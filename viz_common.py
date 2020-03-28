import os
import argparse

from config import parser as base_parser


parser = argparse.ArgumentParser(parents=[base_parser])

parser.add_argument("--weights", "-w", type=str, help="Path to model weights.")
parser.add_argument("--img", type=str,
                    help="Image to run the model on.", required=True)
parser.add_argument("--gray", type=bool, default=True,
                    help="Whether to load the image in grayscale, and apply appropriate model. (default %(default)s)")
parser.add_argument("--save-path", "-o", type=str,
                    help="Save the maps to a file.")

args, extra_args = parser.parse_known_args()

num_channels = 1 if args.gray else 3

_kwargs = {
    'num_channels': num_channels,
    'antialias': args.antialias >= 1,
    'antialias_down_only': args.antialias != 2
}
