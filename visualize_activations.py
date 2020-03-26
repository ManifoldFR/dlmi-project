"""Perform inference on an exemple, and get the resulting activation maps."""
import numpy as np
import torch
from nets import MODEL_DICT
from utils.interpretation import DownBlockActivations
from utils import load_preprocess_image
from utils.plot import plot_with_overlay

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import config
from config import PATCH_SIZE

from viz_common import parser

from typing import List
import json
import argparse

from torchvision.utils import make_grid

with open("dataset_statistics.json") as f:
    dataset_stats_ = json.load(f)


if torch.cuda.is_available():
    from torch.backends import cudnn
    cudnn.benchmark = True
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


parser.prog = "visualize-activations"
parser.description = "Visualize activations in feature maps."

parser.add_argument("--model", choices=MODEL_DICT.keys())
parser.add_argument("--save-path", type=str,
                    help="Save the maps to a file.")

args = parser.parse_args()

num_channels = 1 if args.gray else 3

_kwargs = {
    'num_channels': num_channels,
    'antialias': args.antialias
}

print("Model class: {:s}".format(args.model))
model_cls = MODEL_DICT[args.model]

model = model_cls(**_kwargs)

if args.weights is not None:
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict['model_state_dict'])
else:
    import warnings
    warnings.warn("Model weights not loaded.")
model.to(DEVICE)


img, img_t = load_preprocess_image(args.img, gray=args.gray)
img_t = img_t.to(DEVICE)
input_size = img.shape[:2]
viz_ = DownBlockActivations(model, down_kw=['down1', 'down2', 'down3'])

# Perform prediction

with torch.no_grad():
    prediction_ = model(img_t)
    probas_ = torch.softmax(prediction_, dim=1)
    prediction_ = prediction_.data.cpu().numpy()[0, 1]
    probas_ = probas_.data.cpu().numpy()[0, 1]

# Plotting logic


fig = plot_with_overlay(img, probas_, figsize=(12, 5))


for idx, (name, arr) in enumerate(viz_.get_maps(img_t)):
    print(name, arr.shape, end=' ')
    num_feats_ = arr.shape[1]
    num_rows_mul = num_feats_ // 128
    arr_grid = make_grid(arr.transpose(0, 1), nrow=16, padding=0,
                         normalize=True, scale_each=True)
    arr_grid = arr_grid[0]
    print("grid:", arr_grid.shape)

    fig: plt.Figure = plt.figure(figsize=(10, num_rows_mul * 5), dpi=100)
    ax = fig.add_subplot()
    ims_ = ax.imshow(arr_grid, cmap='viridis')
    ax.set_title("Activations: {:s}".format(name))
    ax.axis('off')
    fig.tight_layout()
    
    if args.save_path is not None:
        fname = "{:s}_{:s}.png".format(args.save_path, name)
        fig.savefig(fname, bbox_inches=None)


plt.show()
