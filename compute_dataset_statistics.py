import torch
from torch.utils.data import ConcatDataset, DataLoader
from utils.datasets import DriveDataset


DRIVE_SUBSET_TRAIN = slice(0, 15)
train_dataset = DriveDataset("data/drive/training", subset=DRIVE_SUBSET_TRAIN)


print(len(train_dataset))



loader = DataLoader(train_dataset, batch_size=8, num_workers=1)

mean = 0.
std = 0.
for images in loader:
    images = images.float() / 255
    # batch size (the last batch can have smaller size!)
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1) * images.size(2), -1)
    mean += images.mean(1).sum(0)
    std += images.std(1).sum(0)

mean /= len(loader.dataset)
std /= len(loader.dataset)

print(mean)
print(std)

import json

with open("dataset_statistics.json", "w+") as f:
    try:
        stats_ = json.load(f)
    except json.JSONDecodeError:
        stats_ = {}

    stats_['drive'] = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    print(stats_)
    json.dump(stats_, f, indent=4)
