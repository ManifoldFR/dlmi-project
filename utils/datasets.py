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
    
    Parameters
    ----------
    transforms
        Applies to both image, mask and target segmentation mask (when available).
    subset : slice
        Subset of indices of the dataset we want to use.
    green_only : bool
        Only use the green channel (idx 1).
    """
    
    def __init__(self, root: str, transforms=None, train: bool=False, subset: slice=None, return_mask=False, green_only=True):
        """
        Parameters
        ----------
        subset : slice
            Slice of data files on which to train.
        """
        super().__init__(root, transforms=transforms)
        self.train = train
        self.use_mask = return_mask
        self.green_only = green_only
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
                if self.green_only:
                    img = img[[1]]
                mask, target = augmented['masks']
                # if isinstance(img, np.ndarray):
                #     img[mask == 0] = 0
                # else:
                #     img[:, mask == 0] = 0
                target = target.astype(int) / 255
            if self.use_mask:
                return img, torch.from_numpy(mask).long(), torch.from_numpy(target).long()
            return img, torch.from_numpy(target).long()
        else:
            if self.transforms is not None:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented['image']
                if self.green_only:
                    img = img[[1]]
                mask = augmented['mask']
                # if isinstance(np.ndarray, img):
                #     img[mask == 0] = 0
                # else:
                #     img[:, mask == 0] = 0
            if self.use_mask:
                return img, torch.from_numpy(mask).long()
            return img


class STAREDataset(VisionDataset):
    """STARE (STructured Analysis of the Retina) retinography dataset
    http://cecas.clemson.edu/~ahoover/stare/.
    
    Parameters
    ----------
    transforms
        Applies to both image, mask and target segmentation mask (when available).
    subset : slice
        Subset of indices of the dataset we want to use.
    green_only : bool
        Only use the green channel (idx 1).
    """
    def __init__(self, root: str, transforms=None, combination_type="random", subset=None, green_only=True):
        super().__init__(root, transforms=transforms)
        self.images = sorted(glob.glob(os.path.join(root, "images/*.ppm")))
        self.target1 = sorted(glob.glob(os.path.join(root, "annotation 1/*.png")))
        self.target2 = sorted(glob.glob(os.path.join(root, "annotation 2/*.png")))
        if subset is not None:
            self.images = self.images[subset]
            self.target1 = self.target1[subset]
            self.target2 = self.target2[subset]
        self.combination_type = combination_type
        self.green_only = green_only

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t1 = cv2.imread(self.target1[index], cv2.IMREAD_UNCHANGED)
        t2 = cv2.imread(self.target2[index], cv2.IMREAD_UNCHANGED)
        target = self.combine_multiple_targets(t1, t2)
        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=target)
            img = augmented['image']
            # pick only green channel
            if self.green_only:
                img = img[[1]]
            target = augmented['mask']
        return img, target

    def combine_multiple_targets(self, t1, t2):
        # TODO implement strategies
        if self.combination_type == "random":
            target=[t1,t2][np.random.randint(2)]

        elif self.combination_type == "union":
            target=(t1+t2>0)*1
    
        elif self.combination_type == "intersection":
            target=((t1==1) & (t2==1))*1

        return target
