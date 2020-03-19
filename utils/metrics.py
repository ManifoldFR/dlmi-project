"""Evaluation metrics."""
import torch
from torch import Tensor
from torch.nn import functional as F
from sklearn.metrics import accuracy_score as _sk_acc

EPSILON = 1e-8


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Hard Intersection-over-Union (IoU) metric.
    
    Source: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    """
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    dims = (1, 2)  # dimensions to sum over
    intersection = (outputs & labels).float().sum(
        dims)  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(
        dims)         # Will be zero if both are 0

    # We smooth our devision to avoid 0/0
    iou = (intersection + EPSILON) / (union + EPSILON)
    return iou


def dice_score(input: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Dice score metric."""
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    input = input.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    dims = (1, 2)  # dimensions to sum over
    # Count zero whenever either prediction or truth = 0
    intersection = (input.float() * labels.float()).sum(dims)
    im_sum = (input + labels).float().sum()

    # We smooth our devision to avoid 0/0
    dice = 2 * intersection / (im_sum + EPSILON)
    return dice

def accuracy(input: torch.Tensor, labels: torch.Tensor) -> float:
    input = input.flatten().detach().cpu().numpy()
    labels = labels.flatten().detach().cpu().numpy()
    acc = _sk_acc(labels, input)
    return acc
