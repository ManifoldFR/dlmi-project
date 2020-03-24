import torch
from torch import nn
import numpy as np
import topologylayer as tl


class TopologicalLoss(nn.Module):
    """Loss function that penalizes absence/presence of topological features.
    See "A Topological Loss Function forDeep-Learning based Image Segmentationusing Persistent Homology"
    arXiv: https://arxiv.org/pdf/1910.01877.pdf 
    """

    def __init__(self, size):
        super().__init__()
        self.layer = tl.nn.LevelSetLayer2D(size)

    def forward(self, prediction: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
        dgm_ = None
        return
