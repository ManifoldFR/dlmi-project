import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nets.unet import AttentionUNet, UNet
from nets import MODEL_DICT
from utils import losses
from utils.plot import plot_prediction
from utils.loaders import train_dataset, val_dataset

torch.random.manual_seed(0)
np.random.seed(0)


LOSSES_DICT = {
    "crossentropy": nn.CrossEntropyLoss(),
    "dice": losses.soft_dice_loss,
    "iou": losses.soft_iou_loss,
    "focal": losses.focal_loss,
    "combined": losses.CombinedLoss()
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=list(MODEL_DICT.keys()),
                    required=True)
parser.add_argument("--loss", type=str, choices=list(LOSSES_DICT.keys()),
                    default="crossentropy")
parser.add_argument("--epochs", "-E", default=40, type=int)
parser.add_argument("--batch-size", "-B", default=1, type=int)
parser.add_argument("--lr", "-lr", default=2e-5, type=float)
parser.add_argument("--validate-every", "-ve", default=2, type=int,
                    help="Validate every X epochs (default %(default)d)")


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, loader: torch.utils.data.DataLoader, criterion, metric, optimizer, epoch, writer: SummaryWriter=None):
    model.train()
    all_loss = []
    all_acc = dict()
    for key in metric:
        all_acc[key] = []
    
    iterator = tqdm.tqdm(enumerate(loader), desc='Training epoch {:d}'.format(epoch))
    for idx, (img, target) in iterator:
        img = img.to(device)
        # print(target.shape)
        # for 1 class, add dim 1
        target.unsqueeze(1)
        target = target.to(device)
        output = model(img)
        loss = criterion(output, target)
        pred = torch.argmax(output, dim=1, keepdim=True)
        
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        all_loss.append(loss.item())
        for key in metric:
            acc = metric[key](pred, target)
            all_acc[key].append(acc.item())
    
    if epoch == 0:
        # only for the first epoch
        writer.add_graph(model, img)
    
    if writer is not None:
        # get dataset's transformer
        transformer = loader.dataset.transforms  # assumption: dataset has transforms attr
        mean = transformer[-2].mean  # assume second-to-last transformer is Normalizer
        std = transformer[-2].std
        fig = plot_prediction(img, output, target, mean, std)
        writer.add_figure("Train/Prediction",  fig, epoch)
    mean_loss = np.mean(all_loss)
    return mean_loss, {key: np.mean(a) for key, a in all_acc.items()}

def validate(model, loader, criterion, metric):
    model.eval()
    with torch.no_grad():
        all_loss = []
        all_acc = dict()
        for key in metric:
            all_acc[key] = []
        for idx, (img, target) in enumerate(loader):
            img = img.to(device)
            # print(target.shape)
            # for 1 class, add dim 1
            target.unsqueeze(1)
            target = target.to(device)
            output = model(img)
            loss = criterion(output, target)
            pred = torch.argmax(output, dim=1, keepdim=True)
            all_loss.append(loss.item())
            
            for key in metric:
                acc = metric[key](pred, target)
                all_acc[key].append(acc.item())
        if writer is not None:
            # get dataset's transformer
            # assumption: dataset has transforms attr
            transformer = loader.dataset.transforms
            # assume second-to-last transformer is Normalizer
            mean = transformer[-2].mean
            std = transformer[-2].std
            fig = plot_prediction(img, output, target, mean, std)
            writer.add_figure("Validation/Prediction",  fig, epoch)
        mean_loss = np.mean(all_loss)
        return np.mean(all_loss), {key: np.mean(a) for key, a in all_acc.items()}


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    print("Using device %s" % device)
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    
    print("Training model %s" % args.model)
    

    # Make model
    model_class = MODEL_DICT[args.model]
    model = model_class(num_classes=2, num_channels=1)  # binary classification
    model = model.to(device)
    
    # Define optimizer and metrics
    print("Learning rate: {:.3g}".format(args.lr))
    print("Using loss {:s}".format(args.loss))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = LOSSES_DICT[args.loss]
    metric = {
        "dice": losses.dice_score,
        "iou": losses.iou_pytorch
    }
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.8)

    # Define TensorBoard summary
    DATASET = "DRIVE"
    comment = "{:s}-{:s}-BatchNorm-{:s}Loss".format(
        DATASET, args.model, args.loss)
    writer = SummaryWriter(comment=comment)
    
    # Define loaders
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
    
    CHECKPOINT_EVERY = 20
    VALIDATE_EVERY = args.validate_every

    for epoch in range(EPOCHS):
        loss, acc = train(model, train_loader, criterion, metric, optimizer, epoch, writer)
        scheduler.step()

        writer.add_scalar("Train/Loss", loss, epoch)
        writer.add_scalar("Train/Dice score", acc["dice"], epoch)
        writer.add_scalar("Train/IoU", acc["iou"], epoch)
        
        if (epoch + 1) % VALIDATE_EVERY == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, metric)
            print("Epoch {:d}: Train loss {:.3g} -- Dice {:.3g} | Validation loss {:.3g} -- Dice {:.3g}".format(
                epoch, loss, acc["dice"], val_loss, val_acc["dice"]))
            writer.add_scalar("Validation/Loss", val_loss, epoch)
            writer.add_scalar("Validation/Dice score", val_acc["dice"], epoch)
            writer.add_scalar("Validation/IoU", val_acc["iou"], epoch)
        
        
        if epoch > 0 and ((epoch+1) % CHECKPOINT_EVERY == 0):
            save_path = "models/%s_%03d.pth" % (comment, epoch)
            print("Saving checkpoint {:s} at epoch {:d}".format(save_path, epoch))
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, save_path)

    writer.close()
