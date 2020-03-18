"""Sanity checks for things like dimensions etc"""
import numpy as np
import nets
from utils.loaders import DATASET_MAP, train_transform


def test_unet():
    net = nets.AttentionUNet(num_channels=1)

    train_dataset = DATASET_MAP['DRIVE']['train']

    train_dataset.use_mask = True  # override
    # tensors, supposedly
    img, mask, target = zip(*[train_dataset[i] for i in range(2)])
    import torch
    from torchvision.utils import make_grid
    
    img = torch.stack(img)
    
    with torch.no_grad():
        pred_mask = net(img)
    print(pred_mask.shape)

    img = make_grid(img)
    pred_mask = make_grid(pred_mask)
    mask = [m.unsqueeze(0) for m in mask]
    mask = make_grid(mask)
    target = [t.unsqueeze(0) for t in target]
    target = make_grid(target)
    import ipdb; ipdb.set_trace()
    
    import matplotlib.pyplot as plt
    
    mean_ = train_transform[-2].mean
    std_ = train_transform[-2].std
    img = np.moveaxis(img.numpy(), 0, -1)
    img = std_[1] * img + mean_[1]
    
    pred_mask = pred_mask.numpy()[0]
    mask = np.moveaxis(mask.numpy(), 0, -1)
    target = target.numpy()[0]
    
    fig = plt.figure(figsize=(4, 7))
    plt.subplot(311)
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(312)
    plt.imshow(pred_mask)
    plt.axis('off')
    
    plt.subplot(313)
    plt.imshow(target)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_unet()
