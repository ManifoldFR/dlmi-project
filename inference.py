import argparse
import os

import numpy as np
import skimage.io
import torch
import tqdm
from torch.utils.data import DataLoader

import torch.nn.functional as F

import nets
from nets import MODEL_DICT
from utils import load_preprocess_image

import config
from config import MODEL_KWARGS

parser = argparse.ArgumentParser(description="Perform inference on a test dataset.")
parser.add_argument("--model", type=str, choices=MODEL_DICT.keys(),
                    help="Model class to use.", required=True)
parser.add_argument("--output", "-o", type=str, help="Output directory path.", required=True)
parser.add_argument("--weights", "-w", required=True, metavar="WEIGHTS_FILE",
                    type=str,
                    help="Model weights.")

device = "cuda" if torch.cuda.is_available() else "cpu"

args = parser.parse_args()


if __name__ == "__main__":
    CHECKPOINT_FILE = args.weights
    print("Checkpoint file: {:s}".format(CHECKPOINT_FILE))
    checkpoint = torch.load(CHECKPOINT_FILE)
    model_state_dict = checkpoint['model_state_dict']
    
    model_class = MODEL_DICT[args.model]
    _kwargs = MODEL_KWARGS.copy()
    
    model = model_class(num_classes=2, **_kwargs)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import glob
    test_data = glob.glob('data/drive/test/images/*.tif')
    test_data = sorted(test_data)

    trange = tqdm.tqdm(test_data)
    trange.set_description(desc="Inference:")
    with torch.no_grad():
        for i, img_path in enumerate(trange):
            orig_img, img  = load_preprocess_image(img_path, gray=True)
            img = img.to(device)
            original_size = orig_img.shape[:2]

            output = model(img).cpu()
            output = F.interpolate(output, size=original_size, mode='bilinear')
            prediction = torch.argmax(output, dim=1, keepdim=True) * 255
            prediction = prediction.byte().numpy()[0]
            # import ipdb; ipdb.set_trace()
            prediction = np.moveaxis(prediction, 0, -1)
            filename = os.path.join(OUTPUT_DIR, "{:d}.png".format(i+1))
            trange.set_postfix_str("Saving result {:s}".format(filename))
            skimage.io.imsave(filename, prediction)
