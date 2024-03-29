"""Various helper and convenience functions"""

import comet_ml
from gridworld import GridWorld
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import pathlib
import matplotlib.pyplot as plt
from itertools import product, chain


class DatasetFromPickle(Dataset):
    """A torch dataset from dataset and, optionally, 'state data' pickle files. No clever batching"""

    def __init__(self, data_path, dtype='bool', state_data_path=None):
        """

        :param data_path: path of the main dataset pickle file
        :param dtype: 'bool' or 'int' data type for the observations
        :param state_data_path: path of the 'state data' pickle file
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        if state_data_path is not None:
            with open(state_data_path, 'rb') as f:
                self.state_data = pickle.load(f)
        else:
            self.state_data = None
        if dtype not in ['int', 'bool']:
            raise ValueError('dtype should be "bool" or "int"')
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
        """Gets a raw numpy observation pair

        :param idx: index of the pair in the dataset
        :return: raw numpy observation pair
        """
        return self.data[idx]

    def get_state_data(self, idx):
        """Fetches a pair of 'state data' tuples

        :param idx: index of the pair in the dataset
        :return: a pair of 'state data' tuples
        """
        if self.state_data is not None:
            return self.state_data[idx]
        else:
            raise Exception('No state data provided')

    def __len__(self):
        return len(self.data)


class BatchIterator:
    """
    An iterator that gives batches from a dataset for the FPS-loss

    Draws random items from the dataset
    """

    def __init__(self, data_path, idx_path, n_trajectories, pairs_per_trajectory, size=None,
                 dtype_for_torch=np.float32):
        """

        :param data_path: path of the main dataset pickle file
        :param idx_path: path of the index pickle file
        :param n_trajectories: number of trajectories used for a batch
        :param pairs_per_trajectory: number of pairs from each trajectory used for a batch
        :param size: number of batches in the iterator (before StopIteration). If not specified, gets set to
            len(data) // (n_trajectories * pairs_per_trajectory)
        :param dtype_for_torch: dtype to cast to (in torch). Should be np.float32 for 'bool' observations and
            int for 'int'
        """
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
        """Refreshes the counter for a new iteration cycle"""
        self.cnt = 0


class EvaluationBatchIterator:
    """
    An iterator that gives batches from a 'synthetic' evaluation dataset

    Draws items from the dataset in order
    """

    def __init__(self, data_path, idx_path, n_negatives, batch_size, dtype_for_torch=np.float32):
        """Dispatches batches of groups of a positive example and several negative examples

        :param data_path: path of the main dataset pickle file
        :param idx_path: path of the index pickle file
        :param n_negatives: number of negatives for each positive in the dataset
        :param batch_size: number of groups in a batch. Total batch size then is (n_negatives + 1) * batch_size
        :param dtype_for_torch: dtype to cast to (in torch). Should be np.float32 for 'bool' observations and
            int for 'int'
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(idx_path, 'rb') as f:
            self.idx = pickle.load(f)
        self.n_negatives = n_negatives
        self.batch_size = batch_size
        self.cnt = 0
        self.seeds = np.array(list(self.idx.keys()), dtype=np.int)
        self.dtype_for_torch = dtype_for_torch
        assert len(self.seeds) % self.batch_size == 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt + self.batch_size <= len(self.seeds):
            self.cnt += self.batch_size
            batch_s = []
            batch_s_prime = []
            batch_seeds = self.seeds[self.cnt - self.batch_size: self.cnt]
            for seed in batch_seeds:
                start, stop = self.idx[seed]
                indices = np.arange(start, stop)
                batch_s += [torch.from_numpy(self.data[i][0].astype(self.dtype_for_torch)) for i in indices]
                batch_s_prime += [torch.from_numpy(self.data[i][1].astype(self.dtype_for_torch)) for i in indices]

            s = torch.stack(batch_s, dim=0)
            s_prime = torch.stack(batch_s_prime, dim=0)
            return (s, s_prime)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.seeds) // self.batch_size

    def refresh(self):
        """Refreshes the counter for a new iteration cycle"""
        self.cnt = 0


# Not used
# class SmallBatchIterator:
#     def __init__(self, data_path, states_ind_path, size=None, cap=100):
#         with open(data_path, 'rb') as f:
#             self.data = pickle.load(f)
#         with open(states_ind_path, 'rb') as f:
#             self.s_ind = pickle.load(f)
#         if size is None:
#             size = len(self.s_ind)
#         self.cnt = 0
#         self.size = size
#         self.cap = cap
#         self.s_tuples = list(self.s_ind.keys())
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.cnt < self.size:
#             self.cnt += 1
#             batch_s = []
#             batch_s_prime = []
#             batch_s_tuple = self.s_tuples[np.random.randint(0, len(self.s_tuples))]
#             pair_inds = self.s_ind[batch_s_tuple]
#             if len(pair_inds) > self.cap:
#                 pair_inds = [pair_inds[i] for i in
#                              np.random.choice(np.arange(0, len(pair_inds)), self.cap, replace=False)]
#             batch_s += [torch.from_numpy(self.data[i][0].astype(np.float32)) for i in pair_inds]
#             batch_s_prime += [torch.from_numpy(self.data[i][1].astype(np.float32)) for i in pair_inds]
#             s = torch.stack(batch_s, dim=0)
#             s_prime = torch.stack(batch_s_prime, dim=0)
#             return (s, s_prime)
#         else:
#             raise StopIteration
#
#     def __len__(self):
#         return self.size
#
#     def refresh(self):
#         self.cnt = 0

def plot_gridworld(n_rows=2, n_cols=3, figsize=(10, 6), eps=0, save_path='gridworld_demo.svg', seed=42, dtype='bool'):
    """Makes a picture of an expert trajectory

    :param n_rows: number of rows to put the grids in
    :param n_cols: number of columns to put the grids in
    :param figsize: figure size
    :param eps: probability of a random action por the expert
    :param save_path: path to save the result
    :param seed: random seed to set to numpy
    :param dtype: observation dtype. For checking that both dtypes work the same way
    """
    total = n_rows * n_cols
    np.random.seed(seed)
    env = GridWorld(5, 5, 3, obs_dtype=dtype)
    env.reset()
    done = False
    grids = [env.render(mode='get_grid')]
    while not done:
        action = env.get_expert_action(eps=eps)
        _, _, done, _ = env.step(action)
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
    """Get the Comet.ml experiment with the specified name and of the specified index
        (out of all experiments with that name)

    Keys should be stored in experiments/comet_keys.pkl
    :param name: name of the Comet.ml experiment
    :param number: list index of the needed experiment
    :param key: if specified, just fetches an experiment with that key, ignoring name and index
    :return:
    """
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
    """collate_fn for correct batch formation for a Dataloader from a DatasetFromPickle dataset"""
    s_batch = torch.stack([obj[0][0] for obj in batch])
    s_prime_batch = torch.stack([obj[0][1] for obj in batch])
    s_data_batch = [obj[1][0] for obj in batch]
    s_prime_data_batch = [obj[1][1] for obj in batch]
    return ((s_batch, s_prime_batch), (s_data_batch, s_prime_data_batch))


def update_and_return(one, other):
    """Returns a copy of the first dict, updated fom the second"""
    dic = one.copy()
    dic.update(other)
    return dic


def get_parameters_list(base_parameters, parameters_to_vary, base_model_parameters, model_parameters_to_vary,
                        mode='gs'):
    """Get a list of parameter kwargs for the train_dssm() function.

    Used for a grid search or a series of experiments.

    :param base_parameters: dict of train_dssm() parameters that are shared for all of the experiments
    :param parameters_to_vary: dict of lists of train_dssm() parameters to be varied
    :param base_model_parameters: dict of model parameters (for train_dssm()) that are shared for all of the experiments
    :param model_parameters_to_vary: dict of lists of model parameters (for train_dssm()) to be varied
    :param mode: 'gs' for grid search or 'series' for a series (for 'series' mode all lists should be the same length)
    :return: list of kwargs dicts for train_dssm()
    """
    itertool = product if mode == 'gs' else zip
    if mode == 'series':
        length = len(next(chain(parameters_to_vary.values(), model_parameters_to_vary.values())))
        for key, value in chain(parameters_to_vary.items(), model_parameters_to_vary.items()):
            assert len(value) == length, f'Incorrect length of {key}, should be {length}'

    if not parameters_to_vary:
        kwargs_list = [base_parameters]
        if mode == 'series':
            kwargs_list = kwargs_list * length
    else:
        names = parameters_to_vary.keys()
        grid_kwargs = [dict(list(zip(names, values))) for values in itertool(*parameters_to_vary.values())]
        kwargs_list = [update_and_return(base_parameters, dic) for dic in grid_kwargs]
    if not model_parameters_to_vary:
        model_kwargs_list = [base_model_parameters]
        if mode == 'series':
            model_kwargs_list = model_kwargs_list * length
    else:
        model_names = model_parameters_to_vary.keys()
        model_grid_kwargs = [dict(list(zip(model_names, values))) for values in
                             itertool(*model_parameters_to_vary.values())]
        model_kwargs_list = [update_and_return(base_model_parameters, dic) for dic in model_grid_kwargs]
    result = []
    for i, (params, model_params) in enumerate(itertool(kwargs_list, model_kwargs_list)):
        all_params = params.copy()
        all_params['model_kwargs'] = model_params
        all_params['comet_tags'].append(mode)
        all_params['experiment_name'] = f'{mode}_{all_params["experiment_name"]}__{i + 1}'
        result.append(all_params)
    return result


def analyze_eval_dataset_button_presses(evaluation_dataset_state_data_path):
    """Computes and prints statistics of numbers of pressed buttons in positives and negatives in a 'synthetic'
        evaluation dataset"""
    with open(evaluation_dataset_state_data_path, 'rb') as f:
        state_data = pickle.load(f)
    print('Len: ', len(state_data))

    positive_trans = np.zeros(4)
    negative_trans = np.zeros(4)
    matr = np.zeros((4, 4))
    correct = 0
    for i in range(0, len(state_data), 5):
        but_s = state_data[i][0][-1]
        but_s_prime = state_data[i][1][-1]
        delta_pos = but_s_prime - but_s
        positive_trans[delta_pos] += 1
        pressed_pos = int(delta_pos > 0)
        pressed_neg = 0
        for j in range(4):
            but_s_prime_neg = state_data[i + 1 + j][1][-1]
            delta_neg = but_s_prime_neg - but_s
            pressed_neg += delta_neg > 0
            negative_trans[delta_neg] += 1
            matr[delta_pos, delta_neg] += 1
        if pressed_pos + pressed_neg > 0:
            correct += pressed_pos / (pressed_pos + pressed_neg)
        else:
            correct += 1 / 5

    print('Positive button presses: ', positive_trans / positive_trans.sum())
    print('Negative button presses: ', negative_trans / negative_trans.sum())
    print('Matrix of this stuff:')
    for row in matr / matr.sum(axis=1, keepdims=True):
        for val in row:
            print(f'{val:.1%}', end=' ')
        print()
    print('Accuracy of "pick random with a pressed button": ', correct / (len(state_data) // 5))
