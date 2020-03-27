import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

from .antialias import Downsample, get_pad_layer


class MaxBlurPool2d(nn.Module):
    """Implementation of Zhang et al [2019]
    "Making Convolutional Networks Shift-Invariant Again"
    arXiv: https://arxiv.org/pdf/1904.11486.pdf
    
    See GitHub repo: https://github.com/adobe/antialiased-cnns 
    """
    
    def __init__(self, kernel_size: int, stride=None, ceil_mode: bool=False,
                 channels=None, filt_size=3):
        super().__init__()
        self.ceil_mode = ceil_mode
        self.channels = channels
        # replace strided MaxPool with stride-1 pooling (which does not downsample) 
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride if stride is not None else kernel_size
        self.base_mp = nn.MaxPool2d(kernel_size=self.kernel_size, stride=1,
                                    ceil_mode=ceil_mode)
        self.ds = Downsample(stride=self.stride, filt_size=filt_size,
                             channels=self.channels)

    def forward(self, input: Tensor) -> Tensor:
        x = self.base_mp(input)
        x = self.ds(x)
        return x


class UpsampleAntialias(nn.Module):
    """Anti-aliased transposed convolution.
    following Zhang et al [2019], and this GitHub issue:
    https://github.com/adobe/antialiased-cnns/issues/28
    """
    
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)),
                          int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [-pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None]*a[None, :])
        filt = filt/torch.sum(filt)
        self.register_buffer(
            'filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class BlurConvTranspose(nn.Module):
    """Antialiased transposed convolution operator.
    
    Drop-in replacement for learnable tranposed convolution `~nn.ConvTranspose2d`,
    which is used with a stride of 2 as a learned upsampling operator in U-Net.
    Here, we perform a transposed convolution with stride 2 as usual, then
    apply the blur.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.base_convt = nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=kernel_size, stride=stride)
        self.ups = UpsampleAntialias(stride=1, channels=out_channels)

    def forward(self, input: Tensor) -> Tensor:
        return self.ups(self.base_convt(input))
