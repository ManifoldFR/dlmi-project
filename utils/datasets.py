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
        transforms: applies to both image, mask and target segmentation mask (when available).
    """
    
    def __init__(self, root: str, transforms=None, train: bool=False, subset: slice=None):
        """
        Parameters
        ----------
        subset : slice
            Slice of data files on which to train.
        """
        super().__init__(root, transforms=transforms)
        self.train = train
        self.images = sorted(glob.glob(os.path.join(root, "images/*.tif")))
        self.masks = sorted(glob.glob(os.path.join(root, "mask/*.gif")))
        if subset is not None:
            self.images = self.images[subset]
            self.masks = self.masks[subset]
        self.targets = None
        if train:
            self.targets = sorted(glob.glob(os.path.join(root, "1st_manual/*.gif")))
            if subset is not None:
                self.targets = self.targets[subset]
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = np.asarray(Image.open(mask_path))
        if self.train:
            tgt_path = self.targets[index]
            target = np.asarray(Image.open(tgt_path))
            if self.transforms is not None:
                augmented = self.transforms(image=img, masks=[mask, target])
                img = augmented['image']
                mask, target = augmented['masks']
                mask_ones = mask == 1
                if isinstance(img, np.ndarray):
                    img[mask_ones] = 0
                else:
                    img[:, mask_ones] = 0
                target = target.astype(int) / 255
            return img, torch.from_numpy(target).long()
        else:
            if self.transforms is not None:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
                mask_ones = mask == 1
                if isinstance(np.ndarray, img):
                    img[mask_ones] = 0
                else:
                    img[:, mask_ones] = 0
            return img


class STAREDataset(VisionDataset):
    """STARE (STructured Analysis of the Retina) retinography dataset
    http://cecas.clemson.edu/~ahoover/stare/.
    
    """
    def __init__(self, root: str, transforms=None):
        super().__init__(root, transforms=transforms)
        self.images = sorted(glob.glob(os.path.join(root, "images/*.ppm")))
        # type of label used is fixed (hard coded)
        self.targets = sorted(glob.glob(os.path.join(root, "labels/labels_vk/*.ppm")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(self.targets[index], cv2.IMREAD_UNCHANGED)
        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=target)
            img = augmented['image']
            target = augmented['mask']
        return img, target
