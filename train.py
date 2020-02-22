import os
import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import losses
from utils.loaders import train_dataset, val_dataset
from nets.unet import UNet, AttentionUNet

import argparse

MODEL_DICT = {
    "unet": UNet,
    "attunet": AttentionUNet
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=list(MODEL_DICT.keys()),
                    required=True)
parser.add_argument("--epochs", "-E", default=10, type=int)
parser.add_argument("--batch-size", "-B", default=1, type=int)
parser.add_argument("--lr", "-lr", default=1e-4, type=float)


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, loader, criterion, optimizer, epoch, writer: SummaryWriter=None):
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
    if writer is not None:
        fig = plot_prediction(img, output, target)
        writer.add_figure("Train/Prediction",  fig)
        
    mean_loss = np.mean(all_loss)
    return mean_loss

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


def plot_prediction(img: torch.Tensor, pred_mask: torch.Tensor, target: torch.Tensor):
    """
    
    """
    import matplotlib.pyplot as plt
    from typing import Tuple
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig: plt.Figure
    img = img.data.numpy()
    pred_mask = pred_mask.data.numpy()
    target = target.data.numpy()
    
    ax1.imshow(img)
    ax1.set_title("Base image")
    ax2.imshow(pred_mask)
    ax2.set_title("Predicted mask")
    ax3.imshow(target)
    ax3.set_title("Real mask")
    
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    print("Using device %s" % device)
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    
    print("Training model %s" % args.model)
    
    # Define TensorBoard summary
    comment = "DRIVE-%s" % args.model
    writer = SummaryWriter(comment=comment)

    # Make model
    model_class = MODEL_DICT[args.model]
    model = UNet(num_classes=2)  # binary classification
    model = model.to(device)
    
    # Define optimizer and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    val_criterion = losses.iou_pytorch  # validation criterion -- iou

    # Define loaders
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, val_criterion)
        print("Epoch {:d}: Loss {:.3g} | Validation loss {:.3g} -- IoU {:.3g}".format(epoch, loss, val_loss, val_acc))

        writer.add_scalar("Train/Loss", loss, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/IoU", val_acc, epoch)

        save_path = "models/%s_drive_%03d.pth" % (args.model, epoch)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, save_path)
    writer.close()    
