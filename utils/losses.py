"""Custom loss functions."""
import torch
from torch import nn, Tensor
from torch.nn import functional as F

EPSILON = 1e-8


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Source: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    """
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0

    # We smooth our devision to avoid 0/0
    iou = (intersection + EPSILON) / (union + EPSILON)
    return iou


def soft_dice_loss(input: torch.Tensor, labels: torch.Tensor, softmax=True) -> torch.Tensor:
    """Mean soft dice loss over the batch.
    
    Parameters
    ----------
    input : Tensor
        (N,C,H,W) Predicted classes for each pixel.
    labels : LongTensor
        (N,C,H,W) Tensor of pixel labels.
    softmax : bool
        Whether to apply `F.softmax` to input to get class probabilities.
    """
    labels = F.one_hot(labels, )
    dims = (1, 2, 3)  # sum over C, H, W
    if softmax:
        input = F.softmax(input, dim=1)
    intersect = torch.sum(input * labels, dim=dims)
    cardinal = torch.sum(input + labels, dim=dims)
    ratio = intersect / (cardinal + EPSILON)
    return torch.mean(1 - 2. * ratio)
