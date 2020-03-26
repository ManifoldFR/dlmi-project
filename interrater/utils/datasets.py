import os.path
import pdb
import glob
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import albumentations.augmentations.functional as F
import pickle as pkl

print("READ DATASETS\n")

class STAREDataset(VisionDataset):
    """STARE (STructured Analysis of the Retina) retinography dataset
    http://cecas.clemson.edu/~ahoover/stare/.
    
    Parameters
    ----------
    transforms
        Transforms into tensor and normalizes
    metrics 
        choose metrics in ["IoU", "scaled_IoU", "entropy", "scaled_entropy"]
    subset : slice
        Subset of indices of the dataset we want to use.
    green_only : bool
        Only use the green channel (idx 1).
    """
    def __init__(self, root: str, transforms=None, metrics = None, subset=None, green_only=True):
        super().__init__(root, transforms=transforms)
        self.target_dict = pkl.load(open(os.path.join(root, "interrater_data",'dict_interrater.pkl'), 'rb'))
#       # only load images for which annotation is available
        self.images = [os.path.join(root,"stare", "images", str(r+".ppm")) for r in list(self.target_dict["stare"]["file_img"])]
        
        # load as tensor the list of interrater metrics for each image
        self.target = torch.tensor(self.target_dict["stare"][metrics])
        
        # Check length of target and image match
        assert len(self.images) == len(self.target)
        
        if subset is not None:
            self.images = [self.images[i] for i in subset]
            self.target = [self.target[i] for i in subset]
        self.green_only = green_only

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        from interrater.config import GAMMA_CORRECTION
        target = self.target[index]
        img_path = self.images[index]
        try : 
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = F.gamma_transform(img, GAMMA_CORRECTION)
            if self.transforms is not None:
                augmented = self.transforms(image=img)
                img = augmented['image']
    
            # pick only green channel (even if no transformation is applied)
            if self.green_only:
                img = img[[1]]
    
            return img, target
        except : 
            pdb.set_trace()
            
            
class ARIADataset(VisionDataset):
    """ARIA retinography dataset
    
    Parameters
    ----------
    transforms
        Applies to both image, mask and target segmentation mask (when available).
    subset : slice
        Subset of indices of the dataset we want to use.
    green_only : bool
        Only use the green channel (idx 1).
    """
    def __init__(self, root: str, transforms=None, metrics = None, subset=None, green_only=True):
        super().__init__(root, transforms=transforms)
        print("root",root)
        self.target_dict = pkl.load(open(os.path.join(root, "interrater_data",'dict_interrater.pkl'), 'rb'))
#       # only load images for which annotation is available
        self.images = [os.path.join(root,"aria", "images", r) for r in list(self.target_dict["aria"]["file_img"])]
        # load as tensor the list of interrater metrics for each image
        self.target = torch.tensor(self.target_dict["aria"][metrics])
        
        print(len(self.images))
        print(len(self.target))
        
        # Check length of target and image match
        assert len(self.images) == len(self.target)

        if subset is not None:
            self.images = [self.images[i] for i in subset]
            self.target = [self.target[i] for i in subset]

        self.green_only = green_only

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        from interrater.config import GAMMA_CORRECTION
        target = self.target[index]
        img_path = self.images[index]
        try : 
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = F.gamma_transform(img, GAMMA_CORRECTION)
            if self.transforms is not None:
                augmented = self.transforms(image=img)
                img = augmented['image']
    
            # pick only green channel (even if no transformation is applied)
            if self.green_only:
                img = img[[1]]
    
            return img, target
        except : 
            pdb.set_trace()