"""General configuration, and other specific configuration options that
are not passed in the CLI."""
import torch
import numpy as np
import argparse

torch.random.manual_seed(0)
np.random.seed(0)

# Default values
# Input image resolution
PATCH_SIZE = 400
GAMMA_CORRECTION = 1.7

parser = argparse.ArgumentParser(add_help=False)


parser.add_argument("--antialias", type=int, default=1, choices=[0, 1, 2],
                    help="Use model with anti-aliased max pooling operator."
                    " 1: only antialiased downsampling."
                    " 2: also antialiases on the upsampling path."
                    " Default: %(default)d")
parser.add_argument("--input-size", '-ins', type=int, default=PATCH_SIZE,
                    help="Network input size. Default: %(default)dpx")

args, _ = parser.parse_known_args()

PATCH_SIZE = args.input_size

DRIVE_SUBSET_TRAIN = slice(0, 15)
DRIVE_SUBSET_VAL = slice(15, 20)

STARE_SUBSET_TRAIN = slice(0, 15)
STARE_SUBSET_VAL = slice(15, 20)

# no subset slices for ARIA -- see the aria_df.csv for its split


# model config
ANTIALIAS = args.antialias

MODEL_KWARGS = {
    "num_channels": 1,
    "antialias": ANTIALIAS >= 1,
    'antialias_down_only': ANTIALIAS != 2
}  # model keyword args
