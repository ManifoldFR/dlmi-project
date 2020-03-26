"""General configuration, and other specific configuration options that
are not passed in the CLI."""
import torch
import numpy as np

torch.random.manual_seed(0)
np.random.seed(0)

DRIVE_SUBSET_TRAIN = slice(0, 15)
DRIVE_SUBSET_VAL = slice(15, 20)

STARE_SUBSET_TRAIN = slice(0, 15)
STARE_SUBSET_VAL = slice(15, 20)

ARIA_SUBSET_TRAIN = slice(0, 107)
ARIA_SUBSET_VAL = slice(107, 143)

# Input image resolution
PATCH_SIZE = 320

GAMMA_CORRECTION = 1.7

# model config
ANTIALIAS = False

MODEL_KWARGS = {
    "antialias": ANTIALIAS,
    "num_channels": 1
}  # model keyword args
