"""Neural networks."""
from .interrater_net import InterraterNet, InterraterNet2, InterraterNet3, InterraterNet4
from .interrater_net import InterraterNet_pool

MODEL_DICT = {
    "InterraterNet": InterraterNet,
    "InterraterNet2": InterraterNet2,
    "InterraterNet3": InterraterNet3,
    "InterraterNet4": InterraterNet4,
    "InterraterNet_pool": InterraterNet_pool
}
