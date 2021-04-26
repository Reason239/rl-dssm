from utils import DatasetFromPickle
from gridworld import GridWorld, get_grid, join_grids, get_colors, is_final, is_starting, n_pressed, get_stage_and_pos
from dssm import DSSM

from collections import Counter, defaultdict
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pathlib
import pickle


def plot_next_from_random(model, n_env, n_top, figsize, seed_base=10000, seed_increment=1, save_path=None, name=None,
                          dtype='bool', n_buttons=3):
    fig, axes = plt.subplots(nrows=n_env, ncols=n_top + 1, figsize=figsize)
    for row in axes:
        for ax in row:
            ax.set_axis_off()
    for i_env in range(n_env):
        env = GridWorld(5, 5, 3, seed=seed_base + i_env * seed_increment)
        env.reset()
        s = env.to_random_state(seed=i_env)
        s_primes = env.get_all_next_states()
        s_model = torch.unsqueeze(torch.from_numpy(s.astype(np.float32)), 0)
        s_prime_model = torch.stack([torch.from_numpy(s_prime.astype(np.float32)) for s_prime in s_primes], dim=0)
        out = torch.squeeze(model((s_model, s_prime_model)))
        ind_sorted = sorted(list(range(len(s_primes))), key=lambda i: out[i], reverse=True)[:n_top]

        ax = axes[i_env, 0]
        ax.imshow(get_grid(s, dtype=dtype, n_buttons=n_buttons))
        ax.set_title('Starting state')

        for j, i_top in enumerate(ind_sorted):
            ax = axes[i_env, j + 1]
            ax.imshow(get_grid(s_primes[i_top], dtype=dtype, n_buttons=n_buttons))
            ax.set_title(f'pred: {out[i_top]:.2f}')

    fig.suptitle('Next state predictions for random states')
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / name)
    plt.show()


if __name__ == '__main__':
    experiment_path_base = pathlib.Path('experiments')
    experiment_name = 'states_1_1000_test4'
    embed_size = 64
    model = DSSM(in_channels=7, height=5, width=5, embed_size=embed_size)
    model.eval()
    model.load_state_dict(torch.load(experiment_path_base / experiment_name / 'best_model.pth'))

    n_env = 10
    n_top = 6
    seed_base = 0
    seed_increment = 0
    name = f'{n_env}_{n_top}_{seed_base}.png'
    save_path = experiment_path_base / experiment_name / 'next_state_prediction'
    plot_next_from_random(model, n_env, n_top, (12, 20), seed_base, seed_increment, save_path, name)
