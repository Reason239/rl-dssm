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

a = torch.randn(2, 3)
print(a / torch.sqrt(torch.sum(a**2, dim=1, keepdim=True)))
