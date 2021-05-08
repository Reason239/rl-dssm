import comet_ml
from utils import get_old_experiment, DatasetFromPickle
from dssm import DSSMEmbed
from gridworld import get_grid, join_grids
import pathlib
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

experiment_path_base = pathlib.Path('experiments')
dataset_path = pathlib.Path('datasets/int_1000')
part = 'train'
experiment_name = 'quant_q100_dist01_c025'
n_z = 100
show_closest = False
save_local = True
save_comet = True
comet_num = -1
comet_exp_key = None
batch_size = 256
embed_size = 64
n_cols = 6
n_rows = 10
figsize = (16, 16)
pixels_per_tile = 10
pixels_between = pixels_per_tile // 2

model = DSSMEmbed(n_z=n_z)
model_path = experiment_path_base / experiment_name / 'best_model.pth'
if not model_path.exists():
    raise Exception(f'No model state dict in {model_path}')
model.load_state_dict(torch.load(model_path))
z_vectors_norm = model.z_vectors_norm
model.eval()

n_buttons = 3
dataset = DatasetFromPickle(dataset_path / f'{part}.pkl', dtype='int')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

if save_local:
    save_path_base = experiment_path_base / experiment_name / 'inspect_embeds'
    save_path_base.mkdir(parents=True, exist_ok=True)
if save_comet:
    comet_experiment = get_old_experiment(experiment_name, comet_num, comet_exp_key)
    print(f'Connected to expirement with key {comet_experiment.get_key()}')


closest = [[] for _ in range(n_z)]
hash_counts = defaultdict(int)

print('Finding closest state transitions...')
for s, s_prime in tqdm(dataloader):
    s_embed = model.embed(s)
    s_prime_embed = model.embed(s_prime)
    embed2 = model.phi2(s_prime_embed - s_embed)
    z_inds = torch.argmax(torch.matmul(embed2, z_vectors_norm.T), dim=1)
    z_matrix = z_vectors_norm[z_inds]
    dist = ((embed2 - z_matrix) ** 2).sum(axis=1)
    for one_s, one_s_prime, z_ind, distance, embed in zip(s, s_prime, z_inds, dist, embed2):
        embed_hash = hash(str(one_s - one_s_prime))
        if embed_hash not in hash_counts:
            closest[z_ind].append(((one_s, one_s_prime), distance, embed_hash))
        hash_counts[embed_hash] += 1

print('Logging pictures...')
closest_sorted = [sorted(arr, key=lambda x: x[1]) for arr in closest]
for z_ind in tqdm(range(n_z)):
    all_pairs = closest_sorted[z_ind]
    n_display = min(n_rows * n_cols, len(all_pairs))
    if show_closest:
        pairs_to_show = all_pairs[:n_display]
    else:
        pairs_to_show = [all_pairs[i] for i in
                         np.linspace(0, len(all_pairs), num=n_display, endpoint=False, dtype=int)]
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    for row in axes:
        for ax in row:
            ax.set_axis_off()
    for i, ((s, s_prime), distance, embed_hash) in enumerate(pairs_to_show):
        ax = axes[i % n_rows, i // n_rows]
        s, s_prime = s.numpy(), s_prime.numpy()
        grid_s = get_grid(s, pixels_per_tile=pixels_per_tile, dtype='int', n_buttons=n_buttons)
        grid_s_prime = get_grid(s_prime, pixels_per_tile=pixels_per_tile, dtype='int', n_buttons=n_buttons)
        grid = join_grids(grid_s, grid_s_prime, pixels_between=pixels_between)
        ax.imshow(grid)
        ax.set_title(f'dist: {distance:.3f}, num: {hash_counts[embed_hash]}')
    total_elements = sum(hash_counts[obj[2]] for obj in all_pairs)
    fig.suptitle(f'Embedding {z_ind}. n_elements: {total_elements}, n_unique: {len(all_pairs)}')
    fig_name = f'embed_{z_ind:02d}'
    if show_closest:
        fig_name += 'closest'
    if save_local:
        plt.savefig(fname=save_path_base / f'{fig_name}.png')
    if save_comet:
        comet_experiment.log_figure(figure_name=fig_name, figure=plt, overwrite=True)
