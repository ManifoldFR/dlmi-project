"""Perform inference on an exemple, and get the resulting activation maps."""
import numpy as np
import torch
from nets import MODEL_DICT
from utils.interpretation import BlockActivations
from utils import load_preprocess_image, preprocess_image
from utils.plot import plot_with_overlay

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

parser.add_argument("--model", choices=MODEL_DICT.keys())

args = parser.parse_args()



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
viz_ = BlockActivations(model, ['down1', 'down2', 'down3'])

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


CHECK_ROBUST = True



def attack_image(img):
    """
    Parameters
    ----------
    img
        Input image.
    """
    from albumentations.augmentations.functional import shift_scale_rotate
    img1 = shift_scale_rotate(img, 0, 1.5, 0, 0)
    img2 = shift_scale_rotate(img, 0, 1.5, .3, -.2)
    
    # plt.subplot(121)
    # plt.imshow(img1)
    
    # plt.subplot(122)
    # plt.imshow(img2)
    # plt.show()
    
    img1_t = preprocess_image(img1).to(DEVICE)
    img2_t = preprocess_image(img2).to(DEVICE)

    maps1_ = viz_.get_maps(img1_t)
    maps2_ = viz_.get_maps(img2_t)
    
    for idx, ((name, arr1), (_, arr2)) in enumerate(zip(maps1_, maps2_)):
        if idx > 0:
            continue
        print(name, arr1.shape, end=' ')
        num_feats_ = arr1.shape[1]
        num_rows_mul = num_feats_ // 128
        arr1_grid = make_grid(arr1.transpose(0, 1), nrow=16, padding=0,
                            normalize=True, scale_each=True)[0]
        arr2_grid = make_grid(arr2.transpose(0, 1), nrow=16, padding=0,
                              normalize=True, scale_each=True)[0]

        fig: plt.Figure = plt.figure(figsize=(8, 8))
        ax = plt.subplot(2,1,1)
        ims_ = ax.imshow(arr1_grid, cmap='viridis')
        ax.set_title("Initial image")

        ax = plt.subplot(2,1,2)
        ims_ = ax.imshow(arr2_grid, cmap='viridis')
        ax.set_title("Shifted image")
        fig.suptitle("Activations: {:s}".format(name))
        
        fig.tight_layout()

if CHECK_ROBUST:

    attack_image(img)
    plt.show()
    
