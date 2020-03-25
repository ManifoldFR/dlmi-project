import torch
from torch import nn, Tensor
import torch.nn.functional as F
from nets import AttentionUNet
from config import PATCH_SIZE

from captum.attr import (
    Deconvolution,
    LayerActivation
)


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
