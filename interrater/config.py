

#import os 
#os.chdir("C:/Users/Philo/Documents/3A -- MVA/DL for medical imaging/retine/dlmi-project")
#os.getcwd()


import torch
import numpy as np

print("READ CONFIG\n")

# import general configuration
#from config import * #doesn't work

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



### Test bools
test_in_train = True
test_in_net = False


# just to handle memory issues
#ratio = 4
ratio = 8
SIZE = int(320/ratio)
MAX_SIZE = int(448/ratio)

### Training params
#epochs = 40
#epochs = 21
epochs = 15
#epochs = 4
#epochs = 10
#batch_size = 5
batch_size = 2
#batch_size = 1

#model = "InterraterNet"
#model = "InterraterNet2"
#model = "InterraterNet3"
#model = "InterraterNet4"
model = "InterraterNet_pool"
num_pool = 1

#lr = 1
lr = 1e-1
#lr = 1e-2
#lr = 5e-3
#lr = 1e-3
#lr = 1e-5
loss = "MSE"

validate_every = 1
dataset = "STARE"

#transforms_name = "None"
#transforms_name = "make_train_transform"
transforms_name = "make_basic_train_transform"

#interrater_metrics = "IoU"
#interrater_metrics = "scaled_IoU"
#interrater_metrics = "scaled_entropy"
interrater_metrics = "entropy"

#import pickle as pkl
#d = pkl.load(open(os.path.join("data", "interrater_data",'dict_interrater.pkl'), 'rb'))



