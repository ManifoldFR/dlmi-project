

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


ARIA_SHUFFLE = np.random.choice(range(143), size=143, replace=False)
STARE_SHUFFLE = np.random.choice(range(21), size=21, replace=False)

STARE_SUBSET_TRAIN = STARE_SHUFFLE[:15]
STARE_SUBSET_VAL = STARE_SHUFFLE[15:21]

# initial config
ARIA_SUBSET_TRAIN = ARIA_SHUFFLE[:107]
print(ARIA_SUBSET_TRAIN)
ARIA_SUBSET_VAL = ARIA_SHUFFLE[107:127]
ARIA_SUBSET_TEST = ARIA_SHUFFLE[127:143]

## change config
#ARIA_SUBSET_TRAIN = slice(20, 127)
#ARIA_SUBSET_VAL = slice(0, 20)
#ARIA_SUBSET_TEST = slice(127, 143)

# Input image resolution
PATCH_SIZE = 320

GAMMA_CORRECTION = 1.2


# just to handle memory issues
#ratio = 1
#ratio = 2
ratio = 4
#ratio = 8
SIZE = int(320/ratio)
MAX_SIZE = int(448/ratio)
print("CONFIG SIZE : ",SIZE,", MAX SIZE ",MAX_SIZE)

### Training params
#epochs=200
epochs = 100
#epochs = 23
#epochs = 15
#epochs = 10
#batch_size = 5
#batch_size = 2
batch_size = 1

#model = "InterraterNet"
model = "InterraterNet_pool"
model_name = model
num_pool = 20
#num_pool = 80
#num_pool = 8
#num_pool = 6
#num_pool = 6
#num_pool = 4
#num_pool = 2
#num_pool = 0

#lr = 1
#lr = 1e-1
#lr = 1e-2
lr = 5e-3
#lr = 1e-3
#lr = 1e-5


#transforms_name = "None"
transforms_name = "make_train_transform"
#transforms_name = "make_basic_train_transform"

interrater_metrics = "IoU"
#interrater_metrics = "entropy"

normalize_metric = True
#normalize_metric = False

if normalize_metric == False :
    interrater_metrics=interrater_metrics
else :
    interrater_metrics+="_norm"

print("metrics used in config : ", interrater_metrics)

loss = "MSE"
loss_name = loss

#normalize_dataset = True
normalize_dataset = False

validate_every = 1
#dataset = "STARE"
dataset = "ARIA"


### Test bools
test_in_train = True
test_in_net = False

sub_val={"ARIA":[ARIA_SUBSET_TRAIN, ARIA_SUBSET_VAL, ARIA_SUBSET_TEST],
         "STARE" : [STARE_SUBSET_TRAIN, STARE_SUBSET_VAL]}

shuffled_DB=True
