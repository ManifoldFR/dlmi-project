"""Utilities."""
from albumentations.pytorch.functional import img_to_tensor
from albumentations.augmentations.functional import gamma_transform
import config
import cv2
import json
from .loaders import get_transforms


with open("dataset_statistics.json") as f:
    dataset_stats_ = json.load(f)


def preprocess_image(img, orig='DRIVE', gray=True):
    _, vt_ = get_transforms(orig)
    gamma_ = config.GAMMA_CORRECTION
    img_t = gamma_transform(img, gamma_)
    img_t = vt_(image=img_t)['image']
    if gray:
        img_t = img_t[[1]].unsqueeze(0)  # shape (B, 1, H, W)

    return img_t

def load_preprocess_image(path, orig='DRIVE', gray=True):
    img_orig = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img_t = preprocess_image(img_orig, orig, gray)
    return img_orig, img_t
