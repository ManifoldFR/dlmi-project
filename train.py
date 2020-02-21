import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.loaders import train_dataset, val_dataset
from nets.unet import UNet, AttentionUNet

import argparse

MODEL_DICT = {
    "unet": UNet,
    "attunet": AttentionUNet
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=list(MODEL_DICT.keys()))
parser.add_argument("--epochs", "-E", default=10, type=int)
parser.add_argument("--batch-size", "-B", default=1, type=int)
parser.add_argument("--lr", "-lr", default=1e-4, type=float)


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, loader, criterion, optimizer):
    model.train()
    all_loss = []
    
    for idx, (img, target) in tqdm.tqdm(enumerate(loader), desc='Training'):
        img = img.to(device)
        # print(target.shape)
        # for 1 class, add dim 1
        target.unsqueeze(1)
        target = target.to(device)
        

        output = model(img)
        loss = criterion(output, target)
        
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        all_loss.append(loss.item())
    return np.mean(all_loss)

def validate(model, loader, criterion, val_criterion):
    model.eval()
    with torch.no_grad():
        all_loss = []
        all_acc = []
        for idx, (img, target) in enumerate(loader):
            img = img.to(device)
            # print(target.shape)
            # for 1 class, add dim 1
            target.unsqueeze(1)
            target = target.to(device)
            output = model(img)
            loss = criterion(output, target)
            pred = torch.argmax(output, dim=1, keepdim=True)
            acc = val_criterion(pred, target)

            all_loss.append(loss.item())
            all_acc.append(acc.item())
        return np.mean(all_loss), np.mean(all_acc)


SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    intersection = (outputs & labels).float().sum(
        (1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(
        (1, 2))         # Will be zzero if both are 0

    # We smooth our devision to avoid 0/0
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    # This is equal to comparing with thresolds
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    # Or thresholded.mean() if you are interested in average across the batch
    return thresholded



if __name__ == "__main__":
    print("Using device %s" % device)
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    print("Training model %s" % args.model)
    model_class = MODEL_DICT[args.model]
    model = UNet(num_classes=2)  # binary class
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    val_criterion = iou_pytorch
    for e in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion, val_criterion)
        print("Epoch {:d}: Loss {:.3g} | Validation loss {:.3g} -- IoU {:.3g}".format(e, loss, val_loss, val_acc))

        save_path = "models/%s_drive_%d.pth" % (args.model, e)

        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, save_path)
    
    
