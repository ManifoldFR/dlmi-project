"""Sanity checks for things like dimensions etc"""
from nets.unet import UNet
from utils.loaders import train_dataset, train_transform


def test_unet():
    net = UNet()

    # tensors, supposedly
    img, target = train_dataset[0]
    img = img.unsqueeze(0)
    print("input shape:", img.shape)
    print("masks shape:", target.shape, target.dtype)

    pred_mask = net(img)
    print(pred_mask.shape)

if __name__ == "__main__":
    test_unet()
