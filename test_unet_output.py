"""Sanity checks for things like dimensions etc"""
import numpy as np
import nets
from utils.loaders import get_datasets
import torch.nn.functional as F


def test_unet():
    net = nets.UNet(num_channels=1)

    # sets_ = get_datasets('DRIVE')
    sets_ = get_datasets('ARIA')
    train_dataset = sets_['train']
    train_dataset.return_mask = True  # override
    # tensors, supposedly
    # img, mask, target = zip(*[train_dataset[i] for i in range(2)])
    img, target = zip(*[train_dataset[i] for i in range(2)])
    mask = None
    import torch
    from torchvision.utils import make_grid
    
    img = torch.stack(img)
    
    with torch.no_grad():
        pred_mask = net(img)
        pred_mask = F.softmax(pred_mask, 1)

    img = make_grid(img)
    pred_mask = make_grid(pred_mask)
    target = make_grid([t.unsqueeze(0) for t in target])
    
    import matplotlib.pyplot as plt
    
    mean_ = train_dataset.transforms[-2].mean
    std_ = train_dataset.transforms[-2].std
    img = (std_[1] * img + mean_[1]).clamp(0, 1)
    img = np.moveaxis(img.numpy(), 0, -1)
    
    if mask is not None:
        mask = make_grid([m.unsqueeze(0) for m in mask])
        mask = np.moveaxis(mask.numpy(), 0, -1)
    
    pred_mask = pred_mask.numpy()[0]
    target = target.numpy()[0]
    
    fig = plt.figure(figsize=(11, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(pred_mask)
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(target)
    plt.axis('off')
    
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_unet()
