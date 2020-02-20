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
        elif norm is None:
            self.bn = nn.Identity()
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


class _UpBlock(nn.Module):
    """Expansive path segment.
    
    Applies `~ConvBlock`, then upsampling deconvolution.
    """

    def __init__(self, in_channels, out_channels, n_convs=2, n_connect=2):
        """
        
        Parameters
        ----------
        n_connect : int
            Multiplicator for the number of input for the 1st convblock after
            the upsampling convolution (useful for skip connections).
        """
        super().__init__()
        layers = [
            # expects multiple of channels
            ConvBlock(n_connect * in_channels, in_channels)
        ] + [
            ConvBlock(in_channels, in_channels)
            for _ in range(n_convs-2)
        ]
        self.conv = nn.Sequential(*layers)
        # counts as one convolution
        self.ups = nn.ConvTranspose2d(in_channels, out_channels,
                                      2, stride=2)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        z = torch.cat((skip, x), dim=1)
        z = self.conv(z)
        out = self.ups(z)  # deconvolve
        return out


class UNet(nn.Module):
    """The U-Net architecture.
    
    See https://arxiv.org/pdf/1505.04597.pdf 
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 2):
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

        self.center = nn.Sequential(
            _DownBlock(512, 1024),
            nn.ConvTranspose2d(1024, 512, 2, stride=2)  # upscale
        )

        # reminder: convolves then upsamples
        self.up1 = _UpBlock(512, 256)
        self.up2 = _UpBlock(256, 128)
        self.up3 = _UpBlock(128, 64)

        self.out_conv = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, num_classes, 3, padding=1)
        )
        self.activation = nn.Softmax(dim=0)

    def forward(self, x: Tensor):
        x1 = self.in_conv(x)  # 64 * 1. * 1. ie 224
        x2 = self.down1(x1)  # 128 * 1/2 * 1/2
        x3 = self.down2(x2)  # 256 * 1/4 * 1/4
        x4 = self.down3(x3)  # 512 * 1/8 * 1/8
        x = self.center(x4)  # 512 * 1/8 * 1/8 ie 28
        x = self.up1(x, x4)  # 256 * 1/4 * 1/4 56
        x = self.up2(x, x3)  # 128 * 1/2 * 1/2 112
        x = self.up3(x, x2)
        z = torch.cat((x1, x), dim=1)
        out = self.activation(self.out_conv(z))
        return out


class AttentionGate(nn.Module):
    """Attention gate for (skip) connections. Produces the attention coefficient
    :math:`alpha`.
    
    See "Attention U-Net:Learning Where to Look for the Pancreas" https://arxiv.org/pdf/1804.03999.pdf
    """

    def __init__(self, gate_channels, feat_channels, int_channels):
        """
        
        Parameters
        ----------
        gate_channels : int
            No. of feature-maps in gate vector.
        feat_channels : int
            No. of feature-maps in lower-level feature vector.
        int_channels : int
            No. of intermediate channels for the attention module.
        """
        super().__init__()
        self.gate_conv = nn.Conv2d(
            gate_channels, int_channels, kernel_size=1, bias=False)
        self.gate_bn = nn.BatchNorm2d(int_channels)
        self.feat_conv = nn.Conv2d(feat_channels, int_channels, kernel_size=1)
        self.feat_bn = nn.BatchNorm2d(int_channels)

        self.alpha_conv = nn.Conv2d(int_channels, 1, kernel_size=1)
        self.alpha_activ = nn.Sigmoid()

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        g
            Gate signal (feature maps from downside block).
        x
            Skip connection weight.
        """
        img_shape = x.shape[-2:]
        g = self.gate_bn(self.gate_conv(g))
        x = self.feat_bn(self.feat_conv(x))
        z = g + x
        z = torch.relu(z)
        z = self.alpha_conv(z)
        alpha = self.alpha_activ(z)
        return alpha


class AttentionUNet(nn.Module):
    """U-Net with attention gates.
    
    See "Attention U-Net:Learning Where to Look for the Pancreas" https://arxiv.org/pdf/1804.03999.pdf
    and original implementation at https://github.com/ozan-oktay/Attention-Gated-Networks.
    """

    def __init__(self, num_channels: int, num_classes: int = 2):
        """
        
        Parameters
        ----------
        gate_channels : int
            No. of feature-maps in gate vector.
        feat_channels : int
            No. of feature-maps in lower-level feature vector.
        int_channels : int
            No. of intermediate channels for the attention module.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.in_conv = ConvBlock(num_channels, 64)

        self.down1 = _DownBlock(64, 128)
        self.down2 = _DownBlock(128, 256)
        self.down3 = _DownBlock(256, 512)

        self.center = nn.Sequential(
            _DownBlock(512, 1024),
            nn.ConvTranspose2d(1024, 512, 2, stride=2)  # upscale
        )

        # reminder: convolves then upsamples
        self.att1 = AttentionGate(512, 512, 128)
        self.up1 = _UpBlock(512, 256)
        self.att2 = AttentionGate(256, 256, 64)
        self.up2 = _UpBlock(256, 128)
        self.att3 = AttentionGate(128, 128, 32)
        self.up3 = _UpBlock(128, 64)

        self.att4 = AttentionGate(64, 64, 16)
        self.out_conv = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, num_classes, 3, padding=1)
        )

        self.activation = nn.Softmax(dim=0)

    def forward(self, x: Tensor):
        x1 = self.in_conv(x)  # 64 * 1. * 1.
        x2 = self.down1(x1)  # 128 * 1/2 * 1/2
        x3 = self.down2(x2)  # 256 * 1/4 * 1/4
        x4 = self.down3(x3)  # 512 * 1/8 * 1/8
        x = self.center(x4)  # 512 * 1/8 * 1/8
        alp1 = self.att1(x4, x4)
        x = self.up1(x, alp1 * x4)  # 256 * 1/4 * 1/4
        alp2 = self.att2(x, x3)
        x = self.up2(x, alp2 * x3)  # 128 * 1/2 * 1/2
        alp3 = self.att3(x, x2)
        x = self.up3(x, alp3 * x2)
        alp4 = self.att4(x, x1)
        z = torch.cat((alp4 * x1, x), dim=1)
        out = self.activation(self.out_conv(z))
        return out
