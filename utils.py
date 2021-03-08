from gridworld import GridWorld, get_grid
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt


class DatasetFromPickle(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        return tuple(torch.from_numpy(s.astype(np.float32)) for s in self.data[idx])

    def get_raw(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class BatchIterator:
    def __init__(self, data_path, idx_path, n_trajectories, pairs_per_trajectory, size=None):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(idx_path, 'rb') as f:
            self.idx = pickle.load(f)
        if size is None:
            size = len(self.data) // (n_trajectories * pairs_per_trajectory)
        self.n_trajectories = n_trajectories
        self.pairs_per_trajectory = pairs_per_trajectory
        self.cnt = 0
        self.size = size
        self.seeds = np.array(list(self.idx.keys()), dtype=np.int)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt < self.size:
            self.cnt += 1
            batch_s = []
            batch_s_prime = []
            batch_seeds = np.random.choice(self.seeds, self.n_trajectories, replace=False)
            for seed in batch_seeds:
                start, stop = self.idx[seed]
                indices = np.random.choice(np.arange(start, stop), self.pairs_per_trajectory, replace=False)
                batch_s += [torch.from_numpy(self.data[i][0].astype(np.float32)) for i in indices]
                batch_s_prime += [torch.from_numpy(self.data[i][1].astype(np.float32)) for i in indices]
            s = torch.stack(batch_s, dim=0)
            s_prime = torch.stack(batch_s_prime, dim=0)
            return (s, s_prime)
        else:
            raise StopIteration

    def __len__(self):
        return self.size

    def refresh(self):
        self.cnt = 0


def format_time(seconds):
    time = round(seconds)
    hours = time // 3600
    minutes = (time % 3600) // 60
    secs = time % 60
    return f'{hours} hours {minutes} minutes {secs} seconds'


def plot_gridworld(n_rows=2, n_cols=3, figsize=(10, 6), eps=0, save_path='gridworld_demo.svg', seed=42):
    total = n_rows * n_cols
    np.random.seed(seed)
    env = GridWorld(5, 5, 3)
    obs = env.reset()
    done = False
    grids = [get_grid(obs)]
    while not done:
        action = env.get_expert_action(eps=eps)
        obs, _, done, _ = env.step(action)
        grids.append(get_grid(obs))
    if total < len(grids):
        display_ind = np.linspace(0, len(grids) - 1, total, dtype=int)
        grids = [grids[i] for i in display_ind]
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    fig.suptitle('Example of an expert trajectory')
    for r in range(n_rows):
        for c in range(n_cols):
            ind = r * n_cols + c
            ax = axes[r, c]
            ax.set_axis_off()
            if ind < len(grids):
                grid = grids[ind]
                ax.imshow(grid)
    plt.savefig(save_path)