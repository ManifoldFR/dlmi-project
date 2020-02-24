import torch
from torch.utils.data import ConcatDataset, DataLoader
from utils.datasets import DriveDataset


train_dataset = DriveDataset("data/drive/training")
test_dataset = DriveDataset("data/drive/test")

full_dataset = ConcatDataset([train_dataset, test_dataset])

print(len(full_dataset))

loader = DataLoader(full_dataset, batch_size=8, num_workers=1)

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
