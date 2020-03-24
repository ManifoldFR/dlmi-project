"""Regularization losses from the paper
"On Regularized Lossesfor Weakly-supervised CNN Segmentation"
http://openaccess.thecvf.com/content_ECCV_2018/papers/Meng_Tang_On_Regularized_Losses_ECCV_2018_paper.pdf 
"""
import torch
from torch import nn
import torch.nn.functional as F


def gaussian_kernel(x: torch.Tensor, bandwidth: float) -> torch.Tensor:
    
    return


def crf_potts_quadratic(input: torch.Tensor) -> torch.Tensor:
    """Potts model quadratic relaxation"""
    return

