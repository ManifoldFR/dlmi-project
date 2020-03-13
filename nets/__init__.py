"""Neural networks."""
from .unet import UNet, AttentionUNet

MODEL_DICT = {
    "unet": UNet,
    "attunet": AttentionUNet
}
