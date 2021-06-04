"""Script to analyze, which next state will be rated best by the model. Not featured in Thesis_v1, because I forgot
    :("""

import comet_ml

from utils import get_old_experiment
from gridworld import GridWorld, grid_from_state_data
from dssm import DSSM, DSSMEmbed

import numpy as np
import torch
import matplotlib.pyplot as plt
import pathlib


def plot_next_from_random(model, n_env, n_top, figsize, seed_base=10000, seed_increment=1, save_path=None,
                          comet_experiment=None, name=None, dtype='bool'):
    """Gets to random states and plot the highest-rated next states for them

    :param model: the model
    :param n_env: number of environments to use
    :param n_top: number of top next states to plot for each environment
    :param figsize: figure size
    :param seed_base: the first seed to use for the environment generation
    :param seed_increment: increment of the seed to generate other environments
    :param save_path: path to save locally
    :param comet_experiment: Comet.ml experiment to log to
    :param name: name of the image file
    :param dtype: observation dtype, 'bool' ot 'int'
    """
    np_dtype = np.float32 if dtype == 'bool' else int
    fig, axes = plt.subplots(nrows=n_env, ncols=n_top + 1, figsize=figsize)
    for row in axes:
        for ax in row:
            ax.set_axis_off()
    for i_env in range(n_env):
        env = GridWorld(5, 5, 3, seed=seed_base + i_env * seed_increment, obs_dtype=dtype)
        env.reset()
        s = env.to_random_state(seed=i_env)
        s_data = env.get_state_data()
        s_primes, s_primes_data = env.get_all_next_states_with_data()
        s_model = torch.unsqueeze(torch.from_numpy(s.astype(np_dtype)), 0)
        s_prime_model = torch.stack([torch.from_numpy(s_prime.astype(np_dtype)) for s_prime in s_primes], dim=0)
        out = torch.squeeze(model((s_model, s_prime_model)))
        ind_sorted = sorted(list(range(len(s_primes))), key=lambda i: out[i], reverse=True)[:n_top]

        ax = axes[i_env, 0]
        ax.imshow(grid_from_state_data(*s_data))
        ax.set_title('Starting state')

        for j, i_top in enumerate(ind_sorted):
            ax = axes[i_env, j + 1]
            ax.imshow(grid_from_state_data(*(s_primes_data[i_top])))
            ax.set_title(f'pred: {out[i_top]:.2f}')

    fig.suptitle('Next state predictions for random states')
    if save_path:
        fig.savefig(save_path / name)
    if comet_experiment:
        comet_experiment.log_figure(figure_name=name, figure=plt, overwrite=True)
    plt.show()


if __name__ == '__main__':
    # Parameters configuration
    experiment_path_base = pathlib.Path('experiments')
    experiment_name = 'reg_test'
    n_z = 50
    save_local = True
    save_comet = True
    comet_num = -1
    comet_exp_key = None
    embed_size = 64
    n_env = 5
    n_top = 15
    seed_base = 0
    seed_increment = 0
    name = f'{n_env}_{n_top}_{seed_base}.png'
    figsize = (2 * n_top + 1, 2 * n_env + 1)
    dtype = 'bool'

    # model = DSSMEmbed(n_z=n_z, embed_size=embed_size)
    model = DSSM(in_channels=7, height=5, width=5, embed_size=embed_size)
    model_path = experiment_path_base / experiment_name / 'best_model.pth'
    if not model_path.exists():
        raise Exception(f'No model state dict in {model_path}')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if save_local:
        save_path = experiment_path_base / experiment_name / 'next_state_prediction'
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = None
    if save_comet:
        comet_experiment = get_old_experiment(experiment_name, comet_num, comet_exp_key)
        print(f'Connected to experiment with key {comet_experiment.get_key()}')
    else:
        comet_experiment = None

    plot_next_from_random(model, n_env, n_top, figsize, seed_base, seed_increment, save_path, comet_experiment, name,
                          dtype)
