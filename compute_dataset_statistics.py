import torch
from torch.utils.data import ConcatDataset, DataLoader
from utils.datasets import DriveDataset, STAREDataset
from config import *

# dataset_name = 'DRIVE'
# train_dataset = DriveDataset("data/drive/training", subset=SUBSET_SLICE)
dataset_name = 'STARE'
train_dataset = STAREDataset("data/stare", subset=STARE_SUBSET_TRAIN)


print(len(train_dataset))

loader = DataLoader(train_dataset, batch_size=8, num_workers=1)

mean = 0.
std = 0.
for images in loader:
    if isinstance(images, list):
        images = images[0]  # only take images
    images = images.float() / 255
    # import ipdb; ipdb.set_trace()
    # batch size (the last batch can have smaller size!)
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1) * images.size(2), -1)
    mean += images.mean(1).sum(0)
    std += images.std(1).sum(0)

mean /= len(loader.dataset)
std /= len(loader.dataset)


import json

try:
    with open("dataset_statistics.json", "r") as f:
        stats_ = json.load(f)
except json.JSONDecodeError:
    stats_ = {}

stats_[dataset_name] = {
    'mean': mean.tolist(),
    'std': std.tolist()
}
print(stats_)

with open("dataset_statistics.json", "w") as f:
    json.dump(stats_, f, indent=4)
