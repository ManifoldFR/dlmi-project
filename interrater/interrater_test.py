"""Perform inference on an exemple, and get the resulting activation maps."""
import numpy as np
import torch
from interrater.nets import MODEL_DICT
from interrater.utils.loaders import *
from interrater.config import ARIA_SUBSET_TEST
import sklearn.metrics as skm

dataset = "ARIA"
transforms_name = "make_basic_train_transform"
interrater_metrics = "IoU_norm"




device = "cuda" if torch.cuda.is_available() else "cpu"


def test(model, loader, criterion, metric):
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
            print(type(img))
            print(img.size())
            target = target.to(device)
            output = model(img)
            loss = criterion(output, target)
            
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



## Faire try : avec 2 modèles possibles et toutes les pool testées 
l=["inter_d26m51s23_019.pth","inter_d27m1s36_019.pth","inter_d27m31s16_019.pth",
   "inter_d27m56s14_199.pth","inter_d28m9s13_099.pth"]
norm_list=[True,False,False,False,False]
results = []
for i in range(len(l)):
    normalize_dataset = norm_list[i]
    DATASET = get_datasets(dataset, transforms_name, interrater_metrics, normalize = normalize_dataset)
    test_dataset = DATASET['test']
    m=torch.load(os.path.join("interrater","models",l[0]))
    mod = MODEL_DICT["InterraterNet_pool"](num_channels=1)
    mod.load_state_dict(state_dict = m["model_state_dict"])
    mod.eval()
    test_loader = DataLoader(test_dataset, 1, shuffle=False)
    test_results = test(mod, 
         test_loader, 
         nn.MSELoss(), 
         metric = {
            "mse" : skm.mean_squared_error,
            "mae" : skm.mean_absolute_error,
            "max_error" : skm.max_error
        })

    results.append(test_results)
    
for i in range(len(l)):
    print(results[i][0])
    print(results[i][1])
    plt.scatter(range(16), results[i][2]["outputs"],label = "output")
    plt.scatter(range(16), results[i][2]["targets"],label = "target")
    plt.legend()
    plt.show()










