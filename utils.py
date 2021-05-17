import comet_ml
from gridworld import GridWorld
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import pathlib
import matplotlib.pyplot as plt
from itertools import product


class DatasetFromPickle(Dataset):
    def __init__(self, data_path, dtype='bool', state_data_path=None):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        if state_data_path is not None:
            with open(state_data_path, 'rb') as f:
                self.state_data = pickle.load(f)
        else:
            self.state_data = None
        if dtype not in ['int', 'bool']:
            raise Exception(f'Incorrect dtype {dtype}. Should be "int" or "bool"')
        self.dtype = dtype

    def __getitem__(self, idx):
        if self.dtype == 'bool':
            value = tuple(torch.from_numpy(s.astype(np.float32)) for s in self.data[idx])
        else:  # self.dtype == 'int'
            value = tuple(torch.from_numpy(s.astype(int)) for s in self.data[idx])
        if self.state_data is not None:
            value = (value, self.get_state_data(idx))
        return value

    def get_raw(self, idx):
        return self.data[idx]

    def get_state_data(self, idx):
        if self.state_data is not None:
            return self.state_data[idx]
        else:
            raise Exception('No state data provided')

    def __len__(self):
        return len(self.data)


class BatchIterator:
    def __init__(self, data_path, idx_path, n_trajectories, pairs_per_trajectory, size=None,
                 dtype_for_torch=np.float32):
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
        self.dtype_for_torch = dtype_for_torch

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt < self.size:
            self.cnt += 1
            batch_s = []
            batch_s_prime = []
            batch_seeds = np.random.choice(self.seeds, self.n_trajectories, replace=False)
            # If there are not enough pairs in a trajectory:
            for _ in range(10):
                if all(self.idx[seed][1] - self.idx[seed][0] >= self.pairs_per_trajectory for seed in batch_seeds):
                    break
                else:
                    batch_seeds = np.random.choice(self.seeds, self.n_trajectories, replace=False)
            else:
                raise Exception(
                    f'Too many retries to find a batch with pairs_per_trajectory={self.pairs_per_trajectory}')
            for seed in batch_seeds:
                start, stop = self.idx[seed]
                indices = np.random.choice(np.arange(start, stop), self.pairs_per_trajectory, replace=False)
                batch_s += [torch.from_numpy(self.data[i][0].astype(self.dtype_for_torch)) for i in indices]
                batch_s_prime += [torch.from_numpy(self.data[i][1].astype(self.dtype_for_torch)) for i in indices]
            s = torch.stack(batch_s, dim=0)
            s_prime = torch.stack(batch_s_prime, dim=0)
            return (s, s_prime)
        else:
            raise StopIteration

    def __len__(self):
        return self.size

    def refresh(self):
        self.cnt = 0


class SmallBatchIterator:
    def __init__(self, data_path, states_ind_path, size=None, cap=100):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(states_ind_path, 'rb') as f:
            self.s_ind = pickle.load(f)
        if size is None:
            size = len(self.s_ind)
        self.cnt = 0
        self.size = size
        self.cap = cap
        self.s_tuples = list(self.s_ind.keys())

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt < self.size:
            self.cnt += 1
            batch_s = []
            batch_s_prime = []
            batch_s_tuple = self.s_tuples[np.random.randint(0, len(self.s_tuples))]
            pair_inds = self.s_ind[batch_s_tuple]
            if len(pair_inds) > self.cap:
                pair_inds = [pair_inds[i] for i in
                             np.random.choice(np.arange(0, len(pair_inds)), self.cap, replace=False)]
            batch_s += [torch.from_numpy(self.data[i][0].astype(np.float32)) for i in pair_inds]
            batch_s_prime += [torch.from_numpy(self.data[i][1].astype(np.float32)) for i in pair_inds]
            s = torch.stack(batch_s, dim=0)
            s_prime = torch.stack(batch_s_prime, dim=0)
            return (s, s_prime)
        else:
            raise StopIteration

    def __len__(self):
        return self.size

    def refresh(self):
        self.cnt = 0


# def format_time(seconds):
#     time = round(seconds)
#     hours = time // 3600
#     minutes = (time % 3600) // 60
#     secs = time % 60
#     return f'{hours} hours {minutes} minutes {secs} seconds'


def plot_gridworld(n_rows=2, n_cols=3, figsize=(10, 6), eps=0, save_path='gridworld_demo.svg', seed=42, dtype='bool'):
    total = n_rows * n_cols
    np.random.seed(seed)
    env = GridWorld(5, 5, 3, obs_dtype=dtype)
    obs = env.reset()
    done = False
    grids = [env.render(mode='get_grid')]
    while not done:
        action = env.get_expert_action(eps=eps)
        obs, _, done, _ = env.step(action)
        grids.append(env.render(mode='get_grid'))
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


def get_old_experiment(name, number=-1, key=None):
    if key is None:
        experiment_keys_path = pathlib.Path('experiments/comet_keys.pkl')
        if not experiment_keys_path.exists():
            raise Exception(f'No experiment_keys file in {experiment_keys_path}')
        with open(experiment_keys_path, 'rb') as f:
            experiment_keys = pickle.load(f)
        keys = experiment_keys[name]
        if not keys:
            raise Exception(f'No saved keys for experiment named {name}')
        key = keys[number]
    experiment = comet_ml.ExistingExperiment(previous_experiment=key, display_summary_level=0)
    return experiment


def my_collate_fn(batch):
    s_batch = torch.stack([obj[0][0] for obj in batch])
    s_prime_batch = torch.stack([obj[0][1] for obj in batch])
    s_data_batch = [obj[1][0] for obj in batch]
    s_prime_data_batch = [obj[1][1] for obj in batch]
    return ((s_batch, s_prime_batch), (s_data_batch, s_prime_data_batch))


def update_and_return(one, other):
    dic = one.copy()
    dic.update(other)
    return dic


def get_kwargs_grid_list(grid, other_kwargs):
    names = grid.keys()
    grid_kwargs = [dict(list(zip(names, values))) for values in product(*grid.values())]
    kwargs_list = [update_and_return(other_kwargs, dic) for dic in grid_kwargs]
    return kwargs_list
