"""Neural networks."""
from .interrater_net import InterraterNet
from .interrater_net import InterraterNet_pool

MODEL_DICT = {
    "InterraterNet": InterraterNet,
    "InterraterNet_pool": InterraterNet_pool
}
