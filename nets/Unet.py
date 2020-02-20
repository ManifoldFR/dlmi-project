import torch
from torch import nn, Tensor


class ConvBlock(nn.Module):
    """Basic convolutional block."""

    def __init__(self, in_channels, out_channels, norm='batch', num_groups=4):
        super().__init__()
        # choice of padding=1 keeps
        # feature map dimensions identical
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(out_channels)
        elif norm == 'group':
            self.bn = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class _DownBlock(nn.Module):
    """Contracting path segment."""

    def __init__(self, in_channels, out_channels, n_convs=3):
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


class _UpBlock(nn.Module):
    """Expansive path segment."""

    def __init__(self, in_channels, out_channels, n_convs=3, n_connect=2):
        """
        
        Parameters
        ----------
        n_connect : int
            Multiplicator for the number of input for the 1st convblock after
            the upsampling convolution (useful for skip connections).
        """
        super().__init__()
        self.ups = nn.ConvTranspose2d(in_channels, out_channels,
                                      2, stride=2)
        layers = [
            ConvBlock(n_connect * out_channels, out_channels)
        ] + [
            ConvBlock(out_channels, out_channels)
            for _ in range(n_convs-2)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.ups(x)  # deconvolve
        z = torch.cat((skip, x), dim=1)
        z = self.conv(z)
        return z


class UNet(nn.Module):
    """The U-Net architecture.
    
    See https://arxiv.org/pdf/1505.04597.pdf 
    """

    def __init__(self, num_channels: int, num_classes: int = 2):
        """Initialize a U-Net.
        
        Parameters
        ----------
        num_channels : int
            Number of input channels.
        num_classes : int
            Number of output classes.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.in_conv = ConvBlock(num_channels, 64)

        self.down1 = _DownBlock(64, 128)
        self.down2 = _DownBlock(128, 256)
        self.down3 = _DownBlock(256, 512)

        self.middle = _DownBlock(512, 1024)

        self.up1 = _UpBlock(1024, 512)
        self.up2 = _UpBlock(512, 256)
        self.up3 = _UpBlock(256, 128)
        self.up4 = _UpBlock(128, 64)

        # binary classification
        self.out_conv = nn.Conv2d(64, num_classes, 3, padding=1)
        self.activation = nn.Softmax(dim=0)

    def forward(self, x: Tensor):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)  # 64
        x3 = self.down2(x2)  # 128
        x4 = self.down3(x3)  # 256
        x5 = self.middle(x4)  # 1024
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.out_conv(x)
        return self.activation(out)


class AttentionUNet(nn.Module):
    """U-Net with attention gates.
    
    See https://arxiv.org/pdf/1804.03999.pdf and original implementation
    at https://github.com/ozan-oktay/Attention-Gated-Networks.
    """

    def __init__(self):
        super().__init__()
