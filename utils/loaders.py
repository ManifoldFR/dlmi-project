"""Dataloaders."""
import torch
import cv2

from utils.datasets import DriveDataset, STAREDataset
from config import *

from albumentations import Compose, Resize, RandomSizedCrop
from albumentations import ElasticTransform
from albumentations import OneOf, Rotate, GaussianBlur, CLAHE, Lambda
from albumentations import VerticalFlip, HorizontalFlip, Resize, Normalize
from albumentations.pytorch import ToTensorV2 as ToTensor

## Define the data augmentation pipeline

MAX_SIZE = 448

def make_train_transform(mean=0, std=1):
    _train = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(90, p=.4, border_mode=cv2.BORDER_CONSTANT, value=0),
        # ElasticTransform(sigma=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=.1),
        OneOf([
            # RandomResizedCrop(PATCH_SIZE, PATCH_SIZE, scale=(.3, 1.), ratio=(1., 1.), p=.8),
            RandomSizedCrop((MAX_SIZE, MAX_SIZE), PATCH_SIZE, PATCH_SIZE, p=.8),
            Resize(PATCH_SIZE, PATCH_SIZE, p=.2),
        ], p=1),
        GaussianBlur(blur_limit=3, p=.2),
        CLAHE(always_apply=True),
        Normalize(mean, std, always_apply=True),
        ToTensor(always_apply=True)
    ])
    return _train


## Use the mean and std values recorded in the JSON file !
# Default train transform converts to Tensor

import json

with open("dataset_statistics.json") as f:
    statistics_ = json.load(f)


train_transform = make_train_transform(
    mean=statistics_['DRIVE']['mean'],
    std=statistics_['DRIVE']['std'])

val_transform = Compose([
    Resize(PATCH_SIZE, PATCH_SIZE),
    CLAHE(always_apply=True),
    Normalize(mean=statistics_['DRIVE']['mean'],
              std=statistics_['DRIVE']['std']),
    ToTensor(),
])

# Test-time data augmentation; we only apply the CLAHE operator and normalize.
test_transform = Compose([
    CLAHE(always_apply=True),
    Normalize(mean=statistics_['DRIVE']['mean'],
              std=statistics_['DRIVE']['std']),
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




DATASET_MAP = {
    "DRIVE": {
        "train": DriveDataset("data/drive/training", transforms=train_transform, 
                              green_only=True, train=True, subset=DRIVE_SUBSET_TRAIN),
        "val": DriveDataset("data/drive/training", transforms=val_transform,
                            green_only=True, train=True, subset=DRIVE_SUBSET_VAL),
        "test": DriveDataset("data/drive/test", transforms=test_transform,
                             green_only=True, train=False)
    },
    "STARE": {
        "train": STAREDataset("data/stare", transforms=train_transform,
                              combination_type="random", subset=STARE_SUBSET_TRAIN),
        "val": STAREDataset("data/stare", transforms=train_transform,
                              combination_type="random", subset=STARE_SUBSET_VAL)
    }
}
