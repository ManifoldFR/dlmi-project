import argparse
import os
os.chdir("C:/Users/Philo/Documents/3A -- MVA/DL for medical imaging/retine/dlmi-project")


import datetime
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.data import DataLoader

#from ignite.contrib.metrics import regression as regmetrics
import sklearn.metrics as skm


#from interrater.config import *
from interrater.config import lr, epochs, batch_size, model, loss, validate_every, num_pool
from interrater.config import dataset, transforms_name, interrater_metrics, test_in_train
from interrater.utils.loaders import *
#from interrater.nets import *
#from interrater.utils.loaders import DATASET_MAP
#from interrater.nets import InterraterNet
from interrater.nets import MODEL_DICT
from interrater.utils.plot import *


torch.random.manual_seed(0)
np.random.seed(0)


LOSSES_DICT = {
    "MSE": nn.MSELoss()
}

#parser = argparse.ArgumentParser()
#parser.add_argument("--model", type=str, choices=list(MODEL_DICT.keys()),
#                    required=True)
#parser.add_argument("--loss", type=str, choices=list(LOSSES_DICT.keys()),
#                    default="MSE")
#parser.add_argument("--dataset", default="STARE", type=str,
#                    help="Specify the dataset.")
#parser.add_argument("--validate-every", "-ve", default=4, type=int,
#                    help="Validate every X epochs (default %(default)d)")
#parser.add_argument("--epochs", "-E", default=40, type=int)
#parser.add_argument("--batch-size", "-B", default=1, type=int)
#parser.add_argument("--lr", "-lr", default=2e-5, type=float)


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, loader: torch.utils.data.DataLoader, criterion, metric, optimizer, epoch):
    model.train()
    all_loss = []
    all_acc = dict()
    for key in metric:
        all_acc[key] = []
    targets_ = []
    outputs_ = []
    
    iterator = tqdm.tqdm(enumerate(loader), desc='Training epoch {:d}'.format(epoch))
    for idx, (img, target) in iterator:
        img = img.to(device)
        target = target.to(device)
        output = model(img)
        loss = criterion(output, target)
#        print("train it")
#        print(target)
#        print(output)
#        print(loss)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        all_loss.append(loss.item())
        targets_ = targets_ + list(target.detach().numpy())
        outputs_ = outputs_ + list(output.detach().numpy())
        
    for key in metric:
        acc = metric[key](np.array(targets_), np.array(outputs_))
        all_acc[key].append(acc)
    
    mean_loss = np.mean(all_loss)
    return mean_loss, {key: np.mean(a) for key, a in all_acc.items()}



###TODO: modify validate
def validate(model, loader, criterion, metric):
    ''' Returns : 
        the mean loss 
        the mean of selected metrics 
        a dictionnary with images, targets and outputs'''
        
    model.eval()
    with torch.no_grad():
        all_loss = []
        all_acc = dict()
        for key in metric:
            all_acc[key] = []
        imgs_ = []
        targets_ = []
        outputs_ = []
        for idx, (img, target) in enumerate(loader):
            img = img.to(device)
            target = target.to(device)
            output = model(img)
            loss = criterion(output, target)
#            print("val it")
#            print(target)
#            print(output)
#            print(loss)
            
            all_loss.append(loss.item())
            # Store the img, target, prediction for plotting
            imgs_.append(img)
            targets_ = targets_ + list(target.detach().numpy())
            outputs_ = outputs_ + list(output.detach().numpy())
        
        for key in metric:
            acc = metric[key](np.array(targets_), np.array(outputs_))
            all_acc[key].append(acc)
        
        mean_loss = np.mean(all_loss)
        
        return mean_loss, {key: np.mean(a) for key, a in all_acc.items()}, {"imgs":imgs_, "targets":targets_, "outputs":outputs_}





if test_in_train == True :
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    
    print("Training model %s" % model)
    
    
    # Make model
    model_class = MODEL_DICT[model]
    
    ##++##++##++##++##++##++##++##++##++##++##++
    ##++ to modify
    ##++##++##++##++##++##++##++##++##++##++##++
    if model == "Interraternet_pool" : 
        model = model_class(num_pool = num_pool, num_channels=1)  
    else :
        model = model_class(num_channels=1)  
    model = model.to(device)
    # Define optimizer and metrics
    print("Learning rate: {:.3g}".format(lr))
    print("Using loss {:s}".format(loss))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = LOSSES_DICT[loss]
    metric_dict = {
        "mse" : skm.mean_squared_error,
        "mae" : skm.mean_absolute_error,
        "max_error" : skm.max_error,
        "explained_var_score" : skm.explained_variance_score
    }
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # Define dataset
    DATASET = dataset
    train_dataset = DATASET_MAP[DATASET]['train']
    val_dataset = DATASET_MAP[DATASET]['val']
    
    # Define loaders
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
    
    CHECKPOINT_EVERY = 20
    VALIDATE_EVERY = validate_every

    save_perf={"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[], "val_details":[]}

    for epoch in range(EPOCHS):
        loss, acc = train(model, train_loader, criterion, metric_dict, optimizer, epoch)
        scheduler.step()
        save_perf["train_loss"].append(loss)
        save_perf["train_acc"].append(acc)
        
        if (epoch + 1) % VALIDATE_EVERY == 0:
            val_loss, val_acc, val_detail = validate(model, val_loader, criterion, metric_dict)
            print("Epoch {:d}: Train loss {:.3g} -- MAE {:.3g} | Validation loss {:.3g} -- MAE {:.3g}".format(
                epoch, loss, acc["mae"], val_loss, val_acc["mae"]))
            save_perf["val_loss"].append(val_loss)
            save_perf["val_acc"].append(val_acc)
            save_perf["val_details"].append(val_detail)
        
        if epoch > 0 and ((epoch+1) % CHECKPOINT_EVERY == 0):
            
            t=datetime.datetime.now()
            save_path = "interrater/models/inter_%s_%03d.pth" % ("d"+str(t.day)+"m"+str(t.minute)+"s"+str(t.second), epoch)
            print("Saving checkpoint {:s} at epoch {:d}".format(save_path, epoch))
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_detail' : val_detail
            }, save_path)


#if __name__ == "__main__":
#    os.makedirs("models", exist_ok=True)
#    print("Using device %s" % device)
#    args = parser.parse_args()
#    
#    EPOCHS = args.epochs
#    BATCH_SIZE = args.batch_size
#    
#    print("Training model %s" % args.model)
#    
#    
#    # Make model
#    model_class = MODEL_DICT[args.model]
#    
#    ##++##++##++##++##++##++##++##++##++##++##++
#    ##++ to modify
#    ##++##++##++##++##++##++##++##++##++##++##++
#    model = model_class(num_channels=1)  
#    model = model.to(device)
#    
#    # Define optimizer and metrics
#    print("Learning rate: {:.3g}".format(args.lr))
#    print("Using loss {:s}".format(args.loss))
#    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#    criterion = LOSSES_DICT[args.loss]
#    metric_dict = {
#        "mse" : skm.mean_squared_error,
#        "mae" : skm.mean_absolute_error,
#        "max_error" : skm.max_error,
#        "explained_var_score" : skm.explained_variance_score,
#        "r2" : skm.r2_score
#    }
#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
#
#    # Define dataset
#    DATASET = args.dataset
#    train_dataset = DATASET_MAP[DATASET]['train']
#    val_dataset = DATASET_MAP[DATASET]['val']
#    
#    # Define loaders
#    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
#    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
#    
#    CHECKPOINT_EVERY = 20
#    VALIDATE_EVERY = args.validate_every
#
#    for epoch in range(EPOCHS):
#        loss, acc = train(model, train_loader, criterion, metric_dict, optimizer, epoch)
#        scheduler.step()
#
#        
#        if (epoch + 1) % VALIDATE_EVERY == 0:
#            val_loss, val_acc, val_detail = validate(model, val_loader, criterion, metric_dict)
#            print("Epoch {:d}: Train loss {:.3g} -- MAE {:.3g} | Validation loss {:.3g} -- MAE {:.3g}".format(
#                epoch, loss, acc["mae"], val_loss, val_acc["mae"]))
#            
#        
#        if epoch > 0 and ((epoch+1) % CHECKPOINT_EVERY == 0):
#            
#            t=datetime.datetime.now()
#            save_path = "interrater/models/inter_%s_%03d.pth" % ("d"+str(t.day)+"m"+str(t.minute)+"s"+str(t.second), epoch)
#            print("Saving checkpoint {:s} at epoch {:d}".format(save_path, epoch))
#            # Save checkpoint
#            torch.save({
#                'epoch': epoch,
#                'model_state_dict': model.state_dict(),
#                'optimizer_state_dict': optimizer.state_dict(),
#                'loss': loss,
#                'val_loss': val_loss,
#                'val_acc': val_acc,
#                'val_detail' : val_detail
#            }, save_path)


print("Final train loss : ", save_perf["train_loss"][epochs-1])
print("Final valid loss : ", save_perf["val_loss"][epochs-1])
print("Ratio between valid and train loss : ", save_perf["val_loss"][epochs-1]/save_perf["train_loss"][epochs-1])


### Plot results
start_ep = 3
plot_loss(save_perf, epochs, validate_every, start_at_epoch = start_ep, save = False, name= "plot_loss", root = "figures")
#plot_metrics("mae",save_perf, epochs, validate_every, start_at_epoch = start_ep, save = False, name= "plot_metric", root = "figures")
#plot_metrics("max_error",save_perf, epochs, validate_every, start_at_epoch = start_ep, save = False, name= "plot_metric", root = "figures")
#plot_target_output(save_perf, metric = interrater_metrics, save = False, name= "plot_target_output", root = "figures")

## Check target VS initial target 
#target = save_perf["val_details"][epochs-1]["targets"]
#output = save_perf["val_details"][epochs-1]["outputs"]
#print("target : ", target)
#print("output : ", output)
#
#
#print("interrater metrics in train file : ",interrater_metrics)


#import pickle as pkl
#target_dict = pkl.load(open(os.path.join("data", "interrater_data",'dict_interrater.pkl'), 'rb'))
#target_dict["stare"][interrater_metrics]









