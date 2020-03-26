import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--weights", help="Path to model weights.")
parser.add_argument("--img", type=str,
                    help="Image to run the model on.", required=True)

parser.add_argument("--gray", type=bool, default=True,
                    help="Whether to load the image in grayscale, and apply appropriate model. (default %(default)s)")

parser.add_argument("--antialias", action='store_true',
                    help="Use model with anti-aliased max pooling operator.")
