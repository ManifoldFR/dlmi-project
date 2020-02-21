import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.loaders import train_dataset, train_transform
from nets.unet import UNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-E", default=10, type=int)
parser.add_argument("--batch-size", "-B", default=2, type=int)
parser.add_argument("--lr", "-lr", default=1e-3, type=float)


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, loader, criterion, optimizer):
    total_loss = 0.
    
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
        
        total_loss += loss.item()
    return total_loss


if __name__ == "__main__":
    print("Using device %s" % device)
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    model = UNet(num_classes=2)  # binary class
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss(reduction='mean')

    for e in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer)
        print("Epoch {:d}: Loss {:.3g}".format(e, loss))
