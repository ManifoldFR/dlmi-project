import os.path
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import torchvision.transforms as tmf
import cv2
from PIL import Image


class DriveDataset(VisionDataset):
    """DRIVE vessel segmentation dataset.
    
    Args:
        transforms: applies to both image and target
    """
    
    def __init__(self, root, transforms=None, transform=None, target_transform=None, train=False):
        super().__init__(root, transforms, transform, target_transform)
    
        self.train = train
        self.images = sorted(glob.glob(os.path.join(root, "images/*.tif")))
        self.masks = sorted(glob.glob(os.path.join(root, "mask/*.gif")))
        self.targets = None
        if train:
            self.targets = sorted(glob.glob(os.path.join(root, "1st_manual/*.gif")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = np.asarray(Image.open(mask_path))
        img = np.dstack([img, mask])

        if self.train:
            tgt_path = self.targets[index]
            target = Image.open(tgt_path)
            return img, np.asarray(target)
        else:
            return img


def test_drive():
    dataset = DriveDataset("data/drive/test", False)
    res = dataset[0]
    img = res[...,:3]
    mask = res[...,3]

    import matplotlib.pyplot as plt
    
    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original image")
    
    plt.subplot(122)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.show()
    return dataset

if __name__ == "__main__":
    dataset = test_drive()
