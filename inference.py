import torch
import nets
from nets import MODEL_DICT

import argparse
from torchvision.datasets import ImageFolder


parser = argparse.ArgumentParser(description="Perform inference on a test dataset.")
parser.add_argument("--model", type=str)
parser.add_argument("--data", type=str, help="Folder for the data.")

device = "cuda" if torch.cuda.is_available() else "cpu"

args = parser.parse_args()

DATA_FOLDER = args.data

dataset = ImageFolder(DATA_FOLDER)


if __name__ == "__main__":
    model_class = MODEL_DICT[args.model]
    model = model_class(num_classes=2)  # binary classification
    model = model.to(device)
