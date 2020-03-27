"""Interpretation of CNNs."""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from nets import AttentionUNet
from config import PATCH_SIZE

from typing import Union, List, Tuple

from captum.attr import (
    LayerActivation
)


class BlockActivations(object):
    
    def __init__(self, model, keys='down'):
        super().__init__()
        """
        Get activations for contracting path of the model.
        """
        self._data = {}
        self.kw = keys
        for name, layer in model.named_children():
            if isinstance(keys, str):
                choice_ = keys in name
            elif isinstance(keys, list):
                choice_ = name in keys
            if choice_:
                self._data[name] = LayerActivation(model.forward, layer)

    def get_maps(self, input: Tensor) -> List[Tuple[str, Tensor]]:
        res_ = []
        for name, activ in self._data.items():
            attr_ = activ.attribute(inputs=input).cpu()
            res_.append((name, attr_))
        return res_

