"""General configuration file."""
import torch
import numpy as np

torch.random.manual_seed(0)
np.random.seed(0)

DRIVE_SUBSET_TRAIN = slice(0, 15)
DRIVE_SUBSET_VAL = slice(15, 23)

STARE_SUBSET_TRAIN = slice(0, 15)
STARE_SUBSET_VAL = slice(15, 21)

ARIA_SUBSET_TRAIN = slice(0, 107)
ARIA_SUBSET_VAL = slice(107, 143)

# Input image resolution
PATCH_SIZE = 320

GAMMA_CORRECTION = 1.2
