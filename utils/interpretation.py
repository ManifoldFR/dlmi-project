import torch
from torch import nn, Tensor
import torch.nn.functional as F
from nets import AttentionUNet
from config import PATCH_SIZE

from typing import Union, List, Tuple

from captum.attr import (
    Deconvolution,
    LayerActivation
)


class DownBlockActivations(object):
    
    def __init__(self, model, down_kw='down'):
        super().__init__()
        """
        Get activations for contracting path of the model.
        """
        self._data = {}
        for name, layer in model.named_children():
            if isinstance(down_kw, str):
                choice_ = down_kw in name
            elif isinstance(down_kw, list):
                choice_ = name in down_kw
            if choice_:
                self._data[name] = LayerActivation(model.forward, layer)

    def get_maps(self, input: Tensor) -> List[Tuple[str, Tensor]]:
        res_ = []
        for name, activ in self._data.items():
            attr_ = activ.attribute(inputs=input).cpu()
            res_.append((name, attr_))
        return res_


class AttentionMapHook:
    """Wrap this around a `~AttentionUNet` model to register a hook that grabs
    the attention maps in a forward pass.
    """
    def __init__(self, model: AttentionUNet, upscale=False, input_size=None):
        super().__init__()
        self._model = model
        self.upscale = upscale  # whether to upscale or not
        self._data = {}
        self._module2name = {}
        if self.upscale and input_size is not None:
            self.size = input_size
        else:
            self.size = (PATCH_SIZE, PATCH_SIZE)
        # Iterate over model children to register hook on the 
        for name, layer in model.named_children():
            if "att" in name:
                print("Registering attention hook on layer {:s}".format(name))
                self._module2name[layer] = name
                layer.register_forward_hook(self._attention_hook)

    def _attention_hook(self, m: nn.Module, input: Tensor, output: Tensor):
        output = F.interpolate(output, size=self.size, mode='bilinear')
        name_ = self._module2name[m]  # layer name
        self._data[name_] = output.data.cpu().numpy()

    def get_maps(self):
        """Get the attention maps. In (name, array) format."""
        return list(self._data.items())
