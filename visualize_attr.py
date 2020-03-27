import numpy as np
import torch
import torch.nn.functional as F
from nets import MODEL_DICT
from utils import load_preprocess_image
from utils.plot import plot_with_overlay

from captum.attr import (
    LayerGradCam
)

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

parser.add_argument("--model", choices=MODEL_DICT.keys(), required=True)


args = parser.parse_args()

num_channels = 1 if args.gray else 3

_kwargs = {
    'num_channels': num_channels,
    'antialias': args.antialias,
    'antialias_down_only': True
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
img_t.requires_grad = True
input_shape = img.shape[:2]

cent_idx = (img_t.shape[0] // 2, img_t.shape[1] // 2)

viz_ = LayerGradCam(model.forward, model.down1)

# network returns tensor of size (N, K, H, W)
attributions = viz_.attribute(img_t, target=(1,) + cent_idx)


# Plot attributions

attr_arr = attributions.data.cpu().numpy()[0, 0]

fig: plt.Figure = plt.figure()
plt.imshow(attr_arr)
plt.title("Attributions for center pixel")
plt.show()

# Perform & plot prediction

prediction_ = model(img_t)
probas_ = torch.softmax(prediction_, dim=1)
probas_ = probas_.data.cpu().numpy()[0, 1]
prediction_ = prediction_.data.cpu().numpy()[0, 1]

## Plotting logic

fig = plot_with_overlay(img, probas_, figsize=(12, 5))
