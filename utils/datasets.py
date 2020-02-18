import os.path
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class DriveDataset(VisionDataset):
    """DRIVE vessel segmentation dataset.
    
    We handle the mask/segmentation mask using the albumentations API, inspired by
    https://github.com/choosehappy/PytorchDigitalPathology/blob/master/segmentation_epistroma_unet/train_unet_albumentations.py
    
    Args:
        transforms: applies to both image and target
    """
    
    def __init__(self, root: str, transforms=None, train=False):
        super().__init__(root, transforms=transforms)
    
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

        if self.train:
            tgt_path = self.targets[index]
            target = Image.open(tgt_path)
            target = np.asarray(target)
            if self.transforms is not None:
                augmented = self.transforms(image=img, masks=[mask, target])
                img = augmented['image']
                mask, target = augmented['masks']
                img[mask == 1] = 0
            return img, target
        else:
            if self.transforms is not None:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
                img[mask == 1] = 0
            return img

if __name__ == "__main__":
    dataset = DriveDataset("data/drive/training", train=True)
    img, target = dataset[0]
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original image")
    
    plt.subplot(122)
    plt.imshow(target, cmap="gray")
    plt.title("Vessel segmentation")
    plt.tight_layout()
    plt.show()
