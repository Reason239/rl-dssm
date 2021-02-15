from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import json


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
