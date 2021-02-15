import numpy as np

import pickle
import torch
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt

from utils import DatasetFromPickle
from gridworld import get_grid, join_grids

data = DatasetFromPickle(data_path='datasets/all_1000/train.pkl')
s, s_prime = data.get_raw(0)
plt.subplot(3, 1, 1)
plt.imshow(get_grid(s))
plt.subplot(3, 1, 2)
plt.imshow(get_grid(s_prime))
plt.subplot(3, 1, 3)
plt.imshow(join_grids(get_grid(s), get_grid(s_prime)))
plt.show()
