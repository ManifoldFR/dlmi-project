"""Dataloaders."""
import torch
import cv2

from utils.datasets import DriveDataset, STAREDataset

from albumentations import Compose, Resize, RandomResizedCrop
from albumentations import OneOf, Rotate, GaussianBlur, CLAHE
from albumentations import VerticalFlip, HorizontalFlip, Resize, Normalize
from albumentations.pytorch import ToTensorV2 as ToTensor

## Define the data augmentation pipeline

def make_train_transform(mean=0, std=1):
    transform_list = [
        Rotate(30, p=.7, border_mode=cv2.BORDER_CONSTANT),
        OneOf([
            RandomResizedCrop(448, 448, scale=(0.4, 1.0), p=.3),
            Resize(448, 448, p=.7),
        ], p=1),
        OneOf([
            HorizontalFlip(),
            VerticalFlip()
        ], p=0.5),
        CLAHE(),
        GaussianBlur(blur_limit=10, p=.3),
        Normalize(mean, std, always_apply=True),
        ToTensor(always_apply=True)
    ]
    return Compose(transform_list)


## The following transforms apply to DRIVE !!
## Use the mean and std values recorded in the JSON file !
# Default train transform converts to Tensor
train_transform = make_train_transform(
    mean=[0.5078, 0.2682, 0.1613],
    std=[0.3378, 0.1753, 0.0978])

val_transform = Compose([
    Resize(448, 448),
    Normalize(mean=[0.5078, 0.2682, 0.1613],
              std=[0.3378, 0.1753, 0.0978]),
    ToTensor(),
])

def denormalize(image: torch.Tensor, normalizer=None, mean=0, std=1):
    """Convert normalized image Tensor to Numpy image array."""
    import numpy as np
    image = np.moveaxis(image.numpy(), 0, -1)
    if normalizer is not None:
        mean = normalizer.mean
        std = normalizer.std
    image = (image * std + mean).clip(0, 1)
    return image


DRIVE_SUBSET_TRAIN = slice(0, 15)
DRIVE_SUBSET_VAL = slice(15, 23)

train_dataset = DriveDataset("data/drive/training",
                             transforms=train_transform,
                             train=True, subset=DRIVE_SUBSET_TRAIN)

val_dataset = DriveDataset("data/drive/training",
                             transforms=val_transform,
                             train=True, subset=DRIVE_SUBSET_VAL)

