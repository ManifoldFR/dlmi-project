import torch
from torch import nn, Tensor

from nets import custom_layers
from .custom_layers import ConvBlock



class _DownBlock(nn.Module):
    """Contracting path segment.
    
    Downsamples using MaxPooling then applies ConvBlock.
    """

    def __init__(self, in_channels, out_channels, n_convs=2, antialias=False):
        super().__init__()
        layers = [
            ConvBlock(in_channels, out_channels)
        ] + [
            ConvBlock(out_channels, out_channels)
            for _ in range(n_convs-1)
        ]
        # maxpooling over patches of size 2
        if not antialias:
            self.mp = nn.MaxPool2d(kernel_size=2)  # stride is equal to kernel_size (2)
        else:
            self.mp = custom_layers.MaxBlurPool2d(kernel_size=2, channels=in_channels)
            # self.mp = MaxBlurPool2d(kernel_size=2)  # ceil mode for consistency
            # self.mp.padding = (1, 1)  # fix some odd problem with dimensionality
        self.conv = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        x = self.mp(input)
        x = self.conv(x)
        return x


class _UpBlock(nn.Module):
    """Expansive path segment.
    
    Applies `~ConvBlock`, then upsampling deconvolution.
    """

    def __init__(self, in_channels, out_channels, n_convs=2, n_connect=2, antialias=False):
        """
        
        Parameters
        ----------
        n_connect : int
            Multiplicator for the number of input for the 1st convblock after
            the upsampling convolution (useful for skip connections).
        """
        super().__init__()
        self.antialias = antialias
        layers = [
            # expects multiple of channels
            ConvBlock(n_connect * in_channels, in_channels)
        ] + [
            ConvBlock(in_channels, in_channels)
            for _ in range(n_convs-1)
        ]
        self.conv = nn.Sequential(*layers)
        # counts as one convolution
        if not self.antialias:
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=2, stride=2)
        else:
            self.upconv = custom_layers.BlurConvTranspose(in_channels, out_channels,
                                                          kernel_size=2, stride=2)

    def forward(self, x: Tensor, skip: Tensor=None) -> Tensor:
        if skip is not None:
            z = torch.cat((skip, x), dim=1)
        else:
            z = x
        z = self.conv(z)
        out = self.upconv(z)  # deconvolve
        return out


class UNet(nn.Module):
    """The U-Net architecture.
    
    See https://arxiv.org/pdf/1505.04597.pdf 
    """

    def __init__(self, num_channels: int=3, num_classes: int=2, antialias=False,
                 antialias_down_only=True):
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
        self.antialias = antialias
        # whether to use blur conv transpose on the upsampling path
        self.antialias_up = self.antialias and not(antialias_down_only)

        self.in_conv = nn.Sequential(
            ConvBlock(num_channels, 64),
            ConvBlock(64, 64)
        )

        self.down1 = _DownBlock(64, 128, antialias=self.antialias)
        self.down2 = _DownBlock(128, 256, antialias=self.antialias)
        self.down3 = _DownBlock(256, 512, antialias=self.antialias)

        self.center = nn.Sequential(
            _DownBlock(512, 1024, antialias=self.antialias),
            nn.ConvTranspose2d(1024, 512, 2, stride=2)  # upscale
        )

        # reminder: convolves then upsamples
        self.up1 = _UpBlock(512, 256, antialias=self.antialias_up)
        self.up2 = _UpBlock(256, 128, antialias=self.antialias_up)
        self.up3 = _UpBlock(128, 64, antialias=self.antialias_up)

        self.out_conv = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )

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
        out = self.out_conv(z)
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
            No. of feature-maps in lower-level feature vector (e.g. skip connection).
        int_channels : int
            No. of intermediate channels for the attention module.
        """
        super().__init__()
        self.feat_channels = feat_channels
        self.gate_channels = gate_channels
        self.int_channels = int_channels
        
        self.gate_conv = nn.Conv2d(gate_channels, int_channels, kernel_size=1, bias=False)
        self.gate_bn = nn.BatchNorm2d(int_channels)
        self.feat_conv = nn.Conv2d(feat_channels, int_channels, kernel_size=1)
        self.feat_bn = nn.BatchNorm2d(int_channels)

        self.alpha_conv = nn.Conv2d(int_channels, 1, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        g
            Gate signal (feature maps from downside block).
        x
            Skip connection input.
        
        Returns
        -------
        Attention map alpha.
        """
        g = self.gate_bn(self.gate_conv(g))
        x = self.feat_bn(self.feat_conv(x))
        z = g + x
        z = torch.relu(z)
        z = self.alpha_conv(z)
        alpha = self.activation(z)
        return alpha


class AttentionUNet(nn.Module):
    """U-Net with attention gates.
    
    See "Attention U-Net:Learning Where to Look for the Pancreas" https://arxiv.org/pdf/1804.03999.pdf
    and original implementation at https://github.com/ozan-oktay/Attention-Gated-Networks.
    """

    def __init__(self, num_channels: int=3, num_classes: int=2, gate_feat_dims: list = None,
                 antialias: bool=False, antialias_down_only=True):
        """
        
        Parameters
        ----------
        gate_channels : int
            No. of feature-maps in gate vector.
        feat_channels : int
            No. of feature-maps in lower-level feature vector.
        int_channels : int
            No. of intermediate channels for the attention module.
        gate_feat_dims : list
            Number of channels for the attention gates.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.antialias = antialias
        # whether to use blur conv transpose on the upsampling path
        self.antialias_up = self.antialias and not(antialias_down_only)
        
        self.in_conv = nn.Sequential(
            ConvBlock(num_channels, 64),
            ConvBlock(64, 64)
        )

        self.down1 = _DownBlock(64, 128, antialias=self.antialias)
        self.down2 = _DownBlock(128, 256, antialias=self.antialias)
        self.down3 = _DownBlock(256, 512, antialias=self.antialias)

        self.center = nn.Sequential(
            _DownBlock(512, 1024, antialias=self.antialias),
            nn.ConvTranspose2d(1024, 512, 2, stride=2)  # upscale
        )

        if gate_feat_dims is None:
            self.gate_feat_dims = [256, 128, 64, 32]
        else:
            assert len(gate_feat_dims) == 4
            self.gate_feat_dims = gate_feat_dims

        # reminder: convolves then upsamples
        self.att1 = AttentionGate(512, 512, self.gate_feat_dims[0])
        self.up1 = _UpBlock(512, 256, self.antialias_up)
        self.att2 = AttentionGate(256, 256, self.gate_feat_dims[1])
        self.up2 = _UpBlock(256, 128, self.antialias_up)
        self.att3 = AttentionGate(128, 128, self.gate_feat_dims[2])
        self.up3 = _UpBlock(128, 64, self.antialias_up)

        self.att4 = AttentionGate(64, 64, self.gate_feat_dims[3])
        self.out_conv = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )

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
        out = self.out_conv(z)
        return out
