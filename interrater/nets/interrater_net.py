import torch
from torch import nn, Tensor
import numpy as np
from interrater.config import *


class ConvBlock(nn.Module):
    """Basic convolutional block."""

    def __init__(self, in_channels, out_channels, norm='batch'):
        super().__init__()
        # choice of padding=1 keeps
        # feature map dimensions identical
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(out_channels)
        elif norm == 'group':
            num_groups = out_channels // 8
            self.bn = nn.GroupNorm(num_groups, out_channels)
        elif norm is None:
            self.bn = nn.Identity()
        else:
            raise TypeError("Wrong type of normalization layer provided for ConvBlock")
        self.activation = nn.ReLU()

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class _DownBlock(nn.Module):
    """Contracting path segment.
    
    Downsamples using MaxPooling then applies ConvBlock.
    """

    def __init__(self, in_channels, out_channels, n_convs=2):
        super().__init__()
        layers = [
            ConvBlock(in_channels, out_channels)
        ] + [
            ConvBlock(out_channels, out_channels)
            for _ in range(n_convs-1)
        ]
        # maxpooling over patches of size 2
        self.mp = nn.MaxPool2d(2)
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mp(x)
        x = self.conv(x)
        return x


class InterraterNet(nn.Module):
    """Interrater network
    """

    def __init__(self, num_channels: int=3):
        """Initialize a U-Net.
        
        Parameters
        ----------
        num_channels : int
            Number of input channels.

        """
        super().__init__()
        self.num_channels = num_channels

        self.in_conv = nn.Sequential(
            ConvBlock(num_channels, 64),
            ConvBlock(64, 64)
        )

        self.down1 = _DownBlock(64, 128)
        self.down2 = _DownBlock(128, 256)
        self.down3 = _DownBlock(256, 512)


    def forward(self, x: Tensor):
        x1 = self.in_conv(x.float())  # 64 * 1. * 1. ie 224
        x2 = self.down1(x1)  # 128 * 1/2 * 1/2
        x3 = self.down2(x2)  # 256 * 1/4 * 1/4
        x4 = self.down3(x3)  # 512 * 1/8 * 1/8
        x4_flatten = x4.view(1, x4.size()[0], -1)
        dim=int(x4.size()[1]*x4.size()[2]*x4.size()[3])
        out = nn.Linear(dim, 1)(x4_flatten)
        out = out.view(x4.size()[0])

        return out
    









