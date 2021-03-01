import numpy as np

import pickle
import torch
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt
# from gridworld import *
from scipy.stats import entropy
import seaborn as sns
# import matplotlib.pylab as plt

a = np.arange(9).reshape((3, 3))
i = np.array([1, 1])
print(a[i[0], i[1]])