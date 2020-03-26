import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F

print("READ NET\n")

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

    def __init__(self, num_channels: int=3, interpolate_dim: int=12800):
        """Initialize a U-Net.
        
        Parameters
        ----------
        num_channels : int
            Number of input channels.

        """
        super().__init__()
        self.num_channels = num_channels
        self.interpolate_dim = interpolate_dim

        self.in_conv = nn.Sequential(
            ConvBlock(num_channels, 64),
            ConvBlock(64, 64)
        )

        self.down1 = _DownBlock(64, 128)
        self.down2 = _DownBlock(128, 256)
        self.down3 = _DownBlock(256, 512)

        self.fc = nn.Linear(self.interpolate_dim ,1)

    def forward(self, x: Tensor):
        x1 = self.in_conv(x.float())  # 64 * 1. * 1. ie 224
        x2 = self.down1(x1)  # 128 * 1/2 * 1/2
        x3 = self.down2(x2)  # 256 * 1/4 * 1/4
        x4 = self.down3(x3)  # 512 * 1/8 * 1/8
        x4_flatten = x4.view(1, x4.size()[0], -1)
        print("flatten size", x4_flatten.size())
        x4_interpolate = F.interpolate(input = x4_flatten, size = self.interpolate_dim)
        out = self.fc(x4_interpolate)
        out = out.view(x4.size()[0])
        return out
    
    
class InterraterNet_pool(nn.Module):
    """Interrater network
    """

    def __init__(self, num_pool: int=3, num_channels: int=3, interpolate_dim: int=12800):
        """Initialize a U-Net.
        
        Parameters
        ----------
        num_channels : int
            Number of input channels.

        """
        super().__init__()
        self.interpolate_dim = interpolate_dim

        self.in_conv = nn.Sequential(
            ConvBlock(num_channels, 64),
            ConvBlock(64, 64)
        )

        self.down1 = _DownBlock(64, 128)
        self.down2 = _DownBlock(128, 256)
        self.down3 = _DownBlock(256, 512)
        
        # Max pooling
#        layers = [ nn.MaxPool1d(kernel_size = 2) for _ in range(num_pool) ]
        layers = [
            nn.MaxPool1d(kernel_size = 2)
        ] + [
            nn.MaxPool1d(kernel_size = 2)
            for _ in range(num_pool-1)
        ]
        self.mp = nn.Sequential(*layers)
#        self.mp = nn.MaxPool1d(kernel_size = 2)
#        self.mp = nn.AvgPool1d(kernel_size = 2)
        
        self.fc = nn.Linear(int(self.interpolate_dim /(2**(num_pool))),1)

    def forward(self, x: Tensor):
#        print(x.size())
        x1 = self.in_conv(x.float())  # 64 * 1. * 1. ie 224
        x2 = self.down1(x1)  # 128 * 1/2 * 1/2
        x3 = self.down2(x2)  # 256 * 1/4 * 1/4
        x4 = self.down3(x3)  # 512 * 1/8 * 1/8
        x4_flatten = x4.view(1, x4.size()[0], -1)
        x4_interpolate = F.interpolate(input = x4_flatten, size = self.interpolate_dim)

        x5 = self.mp(x4_interpolate)
        
        out = self.fc(x5)
        
        out = out.view(x4.size()[0])
        return out



