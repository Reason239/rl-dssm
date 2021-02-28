import numpy as np

import pickle
import torch
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt


a = np.array([True, False, True])
print(np.where(a)[0], np.where(a)[1])