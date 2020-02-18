"""Dataloaders."""
import torch


from datasets import DriveDataset

from albumentations import Compose, Normalize, RandomCrop, Resize
from albumentations import HorizontalFlip, Rotate, GaussianBlur
from albumentations import RandomCrop
from albumentations.pytorch import ToTensor

## Define the data augmentation pipeline

train_transform = Compose([
    Rotate(),    
    GaussianBlur(blur_limit=10),
    RandomCrop(500, 500)
])


train_dataset = DriveDataset("data/drive/training",
                             transforms=train_transform,
                             train=True)


if __name__ == "__main__":    
    import matplotlib.pyplot as plt

    img, target = train_dataset[0]

    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(target, cmap="gray")
    plt.tight_layout()
    plt.show()
