"""Sanity checks for things like dimensions etc"""
import numpy as np
from nets.unet import UNet
from utils.loaders import train_dataset, train_transform


def test_unet():
    net = UNet()

    train_dataset.use_mask = True  # override
    # tensors, supposedly
    img, mask, target = train_dataset[0]
    img = img.unsqueeze(0)
    print("input shape:", img.shape)
    print("masks shape:", target.shape, target.dtype)

    pred_mask = net(img)
    print(pred_mask.shape)
    
    import ipdb; ipdb.set_trace()
    
    import matplotlib.pyplot as plt
    
    img = np.moveaxis(img.numpy()[0], 0, -1)
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(mask.numpy(), cmap="gray")
    plt.subplot(133)
    plt.imshow(target.numpy(), cmap="gray")
    plt.show()

if __name__ == "__main__":
    test_unet()
