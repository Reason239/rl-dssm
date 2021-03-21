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

# plot_gridworld(3, 5, (5, 3), 0, 'test_plot.svg', 0)
l = [(1, 2, 3), (4, 5)]
print(l[np.random.randint(0, len(l))])