import numpy as np

import pickle
import torch
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt
# from gridworld import *
from scipy.stats import entropy
# import matplotlib.pylab as plt
from itertools import combinations
from collections import defaultdict

from utils import plot_gridworld

a = torch.tensor([[1., 2], [3, 4]], requires_grad=True)
b = a[[0, 0, 0, 1]].sum()
b.backward()
print(a.grad)