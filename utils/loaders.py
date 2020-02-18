"""Dataloaders."""
import torch


from datasets import DriveDataset


from albumentations import Compose, Normalize, RandomCrop, Resize
from albumentations import HorizontalFlip, Rotate, GaussianBlur
from albumentations.pytorch import ToTensor

train_transform = Compose([
    HorizontalFlip(always_apply=True),
    GaussianBlur(always_apply=True, blur_limit=10)
])


dataset = DriveDataset("data/drive/training",
                       transforms=train_transform,
                       train=True)

import matplotlib.pyplot as plt

img, target = dataset[0]

plt.figure(figsize=(9, 4))
plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(target, cmap="gray")
plt.tight_layout()
plt.show()
