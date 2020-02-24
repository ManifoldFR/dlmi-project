"""Dataloaders."""
import torch
import cv2

from utils.datasets import DriveDataset, STAREDataset

from albumentations import Compose, Resize, RandomResizedCrop
from albumentations import OneOf, Rotate, GaussianBlur, CLAHE, ElasticTransform
from albumentations import VerticalFlip, HorizontalFlip, Resize, Normalize
from albumentations.pytorch import ToTensorV2 as ToTensor

## Define the data augmentation pipeline

def make_train_transform(to_tensor=False):
    transform_list = [
        Rotate(30, p=.7, border_mode=cv2.BORDER_CONSTANT),
        OneOf([
            RandomResizedCrop(448, 448, scale=(0.3, 1.0), p=.3),
            Resize(448, 448, p=.7),
        ], p=1),
        ElasticTransform(border_mode=cv2.BORDER_CONSTANT, p=.3),
        OneOf([
            HorizontalFlip(),
            VerticalFlip()
        ], p=0.5),
        CLAHE(),
        GaussianBlur(blur_limit=10, p=.3),
    ]
    if to_tensor:
        transform_list += [Normalize(), ToTensor()]
    return Compose(transform_list)


# Default train transform converts to Tensor
train_transform = make_train_transform(True)

val_transform = Compose([
    Resize(448, 448),
    Normalize(),
    ToTensor(),
])

DRIVE_SUBSET_TRAIN = slice(0, 16)
DRIVE_SUBSET_VAL = slice(16, 23)

train_dataset = DriveDataset("data/drive/training",
                             transforms=train_transform,
                             train=True, subset=DRIVE_SUBSET_TRAIN)

val_dataset = DriveDataset("data/drive/training",
                             transforms=val_transform,
                             train=True, subset=DRIVE_SUBSET_VAL)

