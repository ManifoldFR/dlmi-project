import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Decoder(nn.Module):
    """TODO Finish this
    
    Inspired by paper on autoencoder regularization for MRI: https://arxiv.org/pdf/1810.11654.pdf"""
    def __init__(self):
        super().__init__()

