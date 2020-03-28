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

from viz_common import parser, _kwargs

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

print("Model class: {:s}".format(args.model))
model_cls = MODEL_DICT[args.model]

model: torch.nn.Module = model_cls(**_kwargs)

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

# Grab attribution mapping

attr_map_ = {}

for name, layer in model.named_children():
    if "up" in name:
        attr_map_[name] = LayerGradCam(model.forward, model.up3)

data_attr_ = {}
for name, act in attr_map_.items():
    data_attr_[name] = act.attribute(img_t, target=(1,) + cent_idx)


## Plot attributions

fig: plt.Figure = plt.figure(figsize=(12, 4))
for k, (layer_name, attrib) in enumerate(data_attr_.items()):
    plt.subplot(1, 3, k+1)
    attr_arr = attrib.data.cpu().numpy()[0, 0]
    plt.imshow(attr_arr)
    plt.title("GradCam center px %s" % layer_name)
fig.tight_layout()
plt.show()

# Perform & plot prediction

# with torch.no_grad():
#     prediction_ = model(img_t)
#     probas_ = torch.softmax(prediction_, dim=1)
# probas_ = probas_.data.cpu().numpy()[0, 1]
# prediction_ = prediction_.data.cpu().numpy()[0, 1]

# ## Plotting logic

# fig = plot_with_overlay(img, probas_, figsize=(12, 5))
