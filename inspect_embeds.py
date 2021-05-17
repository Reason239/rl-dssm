import comet_ml
from utils import get_old_experiment, DatasetFromPickle, my_collate_fn
from dssm import DSSMEmbed
from gridworld import join_grids, grid_from_state_data, stage_and_pos_from_state_data
import pathlib
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

experiment_path_base = pathlib.Path('experiments')
dataset_path = pathlib.Path('datasets/int_1000')
part = 'train'
experiment_name = 'quant_q50_dist01_c025'
n_z = 50
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
grid_size = (5, 5)

model = DSSMEmbed(n_z=n_z)
model_path = experiment_path_base / experiment_name / 'best_model.pth'
if not model_path.exists():
    raise Exception(f'No model state dict in {model_path}')
model.load_state_dict(torch.load(model_path))
z_vectors_norm = model.z_vectors_norm
model.eval()

n_buttons = 3
dataset = DatasetFromPickle(dataset_path / f'{part}.pkl', dtype='int',
                            state_data_path=dataset_path / f'state_data_{part}.pkl')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=my_collate_fn)

if save_local:
    save_path_base = experiment_path_base / experiment_name / 'inspect_embeds'
    save_path_base.mkdir(parents=True, exist_ok=True)
if save_comet:
    comet_experiment = get_old_experiment(experiment_name, comet_num, comet_exp_key)
    print(f'Connected to expirement with key {comet_experiment.get_key()}')

closest = [[] for _ in range(n_z)]
everything = [[] for _ in range(n_z)]
hash_counts = defaultdict(int)

print('Finding closest state transitions...')
for (s, s_prime), (s_data, s_prime_data) in tqdm(dataloader):
    s_embed = model.embed(s)
    s_prime_embed = model.embed(s_prime)
    embed2 = model.phi2(s_prime_embed - s_embed)
    z_inds = torch.argmax(torch.matmul(embed2, z_vectors_norm.T), dim=1)
    z_matrix = z_vectors_norm[z_inds]
    dist = ((embed2 - z_matrix) ** 2).sum(axis=1)
    for one_s, one_s_prime, one_s_data, one_s_prime_data, z_ind, distance, embed in \
            zip(s, s_prime, s_data, s_prime_data, z_inds, dist, embed2):
        embed_hash = hash(str(one_s - one_s_prime))
        if embed_hash not in hash_counts:
            closest[z_ind].append(((one_s_data, one_s_prime_data), distance, embed_hash))
        everything[z_ind].append(((one_s_data, one_s_prime_data), distance, embed_hash))
        hash_counts[embed_hash] += 1


# s_prime_stages_raw = []
# s_prime_stages_filtered = []
# for (s, s_prime), (s_data, s_prime_data) in tqdm(dataloader):
#     for one_s, one_s_prime, one_s_data, one_s_prime_data in zip(s, s_prime, s_data, s_prime_data):
#         stage_s_prime, pos_s_prime = stage_and_pos_from_state_data(*one_s_prime_data)
#         s_prime_stages_raw.append(stage_s_prime)
#
# for z_ind in range(n_z):
#     for (s_data, s_prime_data), distance, embed_hash in everything[z_ind]:
#         stage_s_prime, pos_s_prime = stage_and_pos_from_state_data(*s_prime_data)
#         s_prime_stages_filtered.append(stage_s_prime)
#
# counts_raw = np.bincount(s_prime_stages_raw)
# counts_raw = counts_raw / counts_raw.sum()
#
# counts_filtered = np.bincount(s_prime_stages_filtered)
# counts_filtered = counts_filtered / counts_filtered.sum()
#
# print(counts_raw)
# print(counts_filtered)

# print('Logging pictures...')
# closest_sorted = [sorted(arr, key=lambda x: x[1]) for arr in closest]
# for z_ind in tqdm(range(n_z)):
#     all_pairs = closest_sorted[z_ind]
#     n_display = min(n_rows * n_cols, len(all_pairs))
#     if show_closest:
#         pairs_to_show = all_pairs[:n_display]
#     else:
#         pairs_to_show = [all_pairs[i] for i in
#                          np.linspace(0, len(all_pairs), num=n_display, endpoint=False, dtype=int)]
#     fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
#     for row in axes:
#         for ax in row:
#             ax.set_axis_off()
#     for i, ((s_data, s_prime_data), distance, embed_hash) in enumerate(pairs_to_show):
#         ax = axes[i % n_rows, i // n_rows]
#         grid_s = grid_from_state_data(*s_data)
#         grid_s_prime = grid_from_state_data(*s_prime_data)
#         grid = join_grids(grid_s, grid_s_prime, pixels_between=pixels_between)
#         ax.imshow(grid)
#         ax.set_title(f'dist: {distance:.3f}, num: {hash_counts[embed_hash]}')
#     total_elements = sum(hash_counts[obj[2]] for obj in all_pairs)
#     fig.suptitle(f'Embedding {z_ind}. n_elements: {total_elements}, n_unique: {len(all_pairs)}')
#     fig_name = f'embed_{z_ind:02d}'
#     if show_closest:
#         fig_name += 'closest'
#     if save_local:
#         plt.savefig(fname=save_path_base / f'{fig_name}.png')
#     if save_comet:
#         comet_experiment.log_figure(figure_name=fig_name, figure=plt, overwrite=True)
#     plt.cla()
#     plt.clf()
#     plt.close(fig)

def create_heat_plot(z_ind, matrix_stage, matrix_pos_s, matrix_pos_s_prime, save_local, save_path_base, save_comet,
                     comet_experiment):
    fig = plt.figure(constrained_layout=True, figsize=(10, 14))
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.set_title(f'Stages for embedding {z_ind}')
    ax1.imshow(matrix_stage, cmap='Blues')
    # ax1.colorbar()

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_title('Pos in s')
    ax2.imshow(matrix_pos_s, cmap='Blues')
    # ax2.colorbar()

    ax3 = fig.add_subplot(gs[2, 1])
    ax3.set_title('Pos in s')
    ax3.imshow(matrix_pos_s_prime, cmap='Blues')
    # ax3.colorbar()
    fig_name = f'e_heat_{z_ind}.png'
    if save_local:
        plt.savefig(fname=save_path_base / f'{fig_name}.png')
    if save_comet:
        comet_experiment.log_figure(figure_name=fig_name, figure=plt, overwrite=True)
    plt.cla()
    plt.clf()
    plt.close(fig)


print('Creating the heatplot')
for z_ind in range(n_z):
    if everything[z_ind]:
        (one_s_data, one_s_prime_data), distance, embed_hash = everything[z_ind][0]
        height, width, n_buttons, _, _, _, _ = one_s_data
        grid_size = (height, width)
        break
matrix_stage_global = np.zeros((3 * n_buttons, 3 * n_buttons))
matrix_pos_s_global = np.zeros(grid_size)
matrix_pos_s_prime_global = np.zeros(grid_size)
for z_ind in tqdm(range(n_z)):
    matrix_stage = np.zeros((3 * n_buttons, 3 * n_buttons))
    matrix_pos_s = np.zeros(grid_size)
    matrix_pos_s_prime = np.zeros(grid_size)
    for (s_data, s_prime_data), distance, embed_hash in everything[z_ind]:
        stage_s, pos_s = stage_and_pos_from_state_data(*s_data)
        stage_s_prime, pos_s_prime = stage_and_pos_from_state_data(*s_prime_data)
        matrix_stage[stage_s, stage_s_prime] += 1
        matrix_pos_s[pos_s[0], pos_s[1]] += 1
        matrix_pos_s_prime[pos_s_prime[0], pos_s_prime[1]] += 1
    matrix_stage_global += matrix_stage
    matrix_pos_s_global += matrix_pos_s
    matrix_pos_s_prime_global += matrix_pos_s_prime
    create_heat_plot(z_ind, matrix_stage, matrix_pos_s, matrix_pos_s_prime, save_local, save_path_base, save_comet,
                     comet_experiment)

create_heat_plot('TOTAL', matrix_stage_global, matrix_pos_s_global, matrix_pos_s_prime_global, save_local,
                 save_path_base, save_comet, comet_experiment)