"""Dataloaders."""
import torch
import cv2

#from interrater.config import *
from interrater.config import interrater_metrics, transforms_name
from interrater.config import STARE_SUBSET_TRAIN, STARE_SUBSET_VAL
from interrater.config import ARIA_SUBSET_TRAIN, ARIA_SUBSET_VAL
from interrater.config import SIZE, MAX_SIZE
from interrater.utils.datasets import STAREDataset

from albumentations import Compose, Resize, RandomSizedCrop
from albumentations import OneOf, Rotate, GaussianBlur, CLAHE
from albumentations import VerticalFlip, HorizontalFlip, Resize, Normalize
from albumentations.pytorch import ToTensorV2 as ToTensor

import json

#os.chdir("C:/Users/Philo/Documents/3A -- MVA/DL for medical imaging/retine/dlmi-project/interrater")
#os.getcwd()


print("READ LOADER\n")
print("interrater metrics in loaders : ", interrater_metrics)



#SIZE = 320
#MAX_SIZE = 448

def make_train_transform(mean=0, std=1):
    print("\ntrain transform with SIZE=",SIZE," and MAX_SIZE = ",MAX_SIZE,"\n")
    _train = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        GaussianBlur(blur_limit=3, p=.2),
        Rotate(45, p=.7, border_mode=cv2.BORDER_CONSTANT),
        OneOf([
            RandomSizedCrop((MAX_SIZE, MAX_SIZE), SIZE, SIZE, p=.8),
            Resize(SIZE, SIZE, p=.2),
        ], p=1),
        CLAHE(always_apply=True),
        Normalize(mean, std, always_apply=True),
        ToTensor(always_apply=True)
    ])
    return _train


def make_basic_train_transform(mean=0, std=1):
    print("\nbasic transform\n")
    _train = Compose([
        OneOf([
            RandomSizedCrop((MAX_SIZE, MAX_SIZE), SIZE, SIZE, p=.8),
            Resize(SIZE, SIZE, p=.2),
        ], p=1),
        CLAHE(always_apply=True),
        Normalize(mean, std, always_apply=True),
        ToTensor(always_apply=True)
    ])
    return _train



## Use the mean and std values recorded in the JSON file !
# Default train transform converts to Tensor

with open("dataset_statistics.json") as f:
    statistics_ = json.load(f)

train_transform = make_train_transform(
    mean=statistics_['STARE']['mean'],
    std=statistics_['STARE']['std'])


transforms_dict={"None":None, 
                 "make_train_transform":make_train_transform(),
                 "make_basic_train_transform":make_basic_train_transform()}
config_transform=transforms_dict[transforms_name] #import transformation from config


DATASET_MAP = {
    "STARE": {
        "train": STAREDataset("data/", transforms=config_transform,
                              metrics=interrater_metrics, subset=STARE_SUBSET_TRAIN),
        "val": STAREDataset("data/", transforms=config_transform,
                              metrics=interrater_metrics, subset=STARE_SUBSET_VAL)
    },
    "ARIA": {
    "train": STAREDataset("data/", transforms=config_transform,
                          metrics=interrater_metrics, subset=ARIA_SUBSET_TRAIN),
    "val": STAREDataset("data/", transforms=config_transform,
                          metrics=interrater_metrics, subset=ARIA_SUBSET_VAL)
    }
}

    

