import torch
from torch import nn, Tensor


class UNet(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        pass
