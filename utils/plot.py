"""Plotting utilities."""
import torch
import torch.nn.functional as F
import numpy as np
from .loaders import denormalize

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
    batch_size = img.shape[0]
    img = make_grid(img, 4)
    # put on CPU, denormalize
    img = denormalize(img.data.cpu(), mean=mean, std=std)
    if apply_softmax:
        pred_mask = F.softmax(pred_mask, dim=1)  # actually apply Softmax
    pred_mask = make_grid(pred_mask, 4)
    if target is not None:
        num_plots = 3
        target = make_grid(target, 4)
        if target.ndim == 3:
            # put in format (B, C, H, W) i.e. add the channel dimension
            target = target.unsqueeze(1)
        target = make_grid(target, 4)
        target = target.detach().cpu().numpy()  # collapse useless dimension
    else:
        num_plots = 2
    pred_mask = F.softmax(pred_mask.data.cpu(), dim=1).numpy()
    pred_mask = pred_mask[1]  # class 1
    
    norm = colors.PowerNorm(0.5, vmin=0., vmax=1., clip=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, num_plots,
                                        figsize=(4*num_plots, 5), dpi=60)
    fig: plt.Figure

    ax1.imshow(img)
    ax1.set_title("Base image")
    ax2.imshow(pred_mask, norm=norm)
    ax2.set_title("Mask probability map")
    ax3.imshow(target, cmap="gray")
    ax3.set_title("Real mask")

    fig.tight_layout()
    return fig
