"""Plotting utilities."""
import torch
import torch.nn.functional as F
import numpy as np
from .loaders import denormalize

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Tuple


def plot_prediction(img: torch.Tensor, pred_mask: torch.Tensor, target: torch.Tensor, mean, std, writer: SummaryWriter = None, apply_softmax=True):
    """Plot the original image, heatmap of predicted class probabilities, and target mask.
    
    Parameters
    ----------
    We expect the inputs to be 4D mini-batch `Tensor`s of shape (B x C x H x W) (except for target which can be B x H x W and is handled in that case).
    The `img` image tensor is expected to be in cv2 `(B,G,R)` format. `pred_mask` is expected to be pre-Softmax unless `apply_softmax` is True.
    """
    batch_size = min(12, img.shape[0])  # never plot more than 2 images
    ncol = batch_size // 4
    img = make_grid(img, 4)
    # put on CPU, denormalize
    # GREEN MODE
    mean = mean[1]
    std = std[1]
    img = denormalize(img.data.cpu(), mean=mean, std=std)
    if target is not None:
        num_plots = 3
        if target.ndim == 3:
            # put in format (B, C, H, W) i.e. add the channel dimension
            target = target.unsqueeze(1)
        target = make_grid(target, 4)
        target = target.detach().cpu().numpy()
        # collapse useless dimension
        target = target[0]
    else:
        num_plots = 2
    
    if apply_softmax:
        pred_mask = F.softmax(pred_mask.data.cpu(), dim=1)
    pred_mask = make_grid(pred_mask, 4).numpy()
    pred_mask = pred_mask[1]  # class 1
    
    norm = colors.PowerNorm(0.5, vmin=0., vmax=1., clip=True)
    splt_nums = (1, num_plots)
    fig, axes = plt.subplots(*splt_nums, figsize=(6*num_plots+1, 5), dpi=60)
    fig: plt.Figure
    (ax1, ax2, ax3) = axes

    ax1.imshow(img)
    ax1.set_title("Base image")
    ax1.axis('off')
    
    ax2.imshow(pred_mask, norm=norm)
    ax2.set_title("Mask probability map")
    ax2.axis('off')
    
    ax3.imshow(target, cmap="gray")
    ax3.set_title("Real mask")
    ax3.axis('off')

    fig.tight_layout()
    return fig
