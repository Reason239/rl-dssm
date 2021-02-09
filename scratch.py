import numpy as np

import pickle
import torch
from tqdm import tqdm
import time
import json

idx_path = 'datasets/all_1000/idx_train.json'
with open(idx_path, 'r') as f:
    idx = json.load(f)
print(type(list(idx.keys())[0]))