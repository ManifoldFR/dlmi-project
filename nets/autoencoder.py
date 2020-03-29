"""Variational autoencoder."""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Normal

from .unet import UNet
from .custom_layers import ConvBlock


class VAEBlock(nn.Module):
    """
    
    The forward method samples from the approximate variational posterior
    in a way that is differentiable.
    """
    
    def __init__(self, in_channels):
        """
        Parameters
        ----------
        in_channels
            Number of input channels provided by the encoding path.
        """
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16)
        )


    def forward(self, input: Tensor) -> Tensor:
        params_ = self.conv(input)
        mu  = params_[..., :128]
        std = params_[..., 128:]
        n = Normal(mu, std)
        z = n.rsample()  # latent variable
        return z
        

class AutoEncoder(nn.Module):
    """AutoEncoder based on U-Net."""
    
    def __init__(self, base_model: UNet):
        """
        Parameter
        ---------
        base_model
            Backend U-Net instance.
        """
        self.base_model = base_model
        # by
        # now register a hook


