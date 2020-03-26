"""Utilities."""
from albumentations.pytorch.functional import img_to_tensor
from albumentations.augmentations.functional import resize, normalize, gamma_transform
from config import PATCH_SIZE, GAMMA_CORRECTION
import cv2
import json

with open("dataset_statistics.json") as f:
    dataset_stats_ = json.load(f)



def load_preprocess_image(path, orig='DRIVE', gray=True):
    mean_ = dataset_stats_[orig]['mean']
    std_ = dataset_stats_[orig]['std']
    img_orig = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img_t = gamma_transform(img_orig, GAMMA_CORRECTION)
    img_t = resize(img_t, PATCH_SIZE, PATCH_SIZE)
    img_t = normalize(img_t, mean_, std_)
    img_t = img_to_tensor(img_t)
    if gray:
        img_t = img_t[[1]].unsqueeze(0)  # shape (B, 1, H, W)
    return img_orig, img_t
