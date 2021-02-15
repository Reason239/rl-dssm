from utils import DatasetFromPickle
from gridworld import get_grid, join_grids
from dssm import DSSM

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
import pathlib

dataset_path = 'datasets/new_1000/'
experiment_path_base = 'experiments/'
experiment_name = 'new_16_4_l'
save_path = pathlib.Path(experiment_path_base + experiment_name + '/clustering2')
save_path.mkdir(parents=True, exist_ok=True)
batch_size = 256
embed_size = 64
n_clusters = 100
n_cols = 6
n_rows = 10
figsize = (14, 14)
pixels_per_tile = 10
pixels_between = pixels_per_tile // 2

train_dataset = DatasetFromPickle(dataset_path + 'train.pkl')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

model = DSSM(in_channels=7, height=5, width=5, embed_size=embed_size)
model.eval()
model.load_state_dict(torch.load(experiment_path_base + experiment_name + '/best_model.pth'))

embeds = []
with torch.no_grad():
    for s, s_prime in tqdm(train_dataloader):
        embeds.append(model.phi2(s_prime - s).numpy())

x_raw = np.concatenate(embeds)
print(f'Initial data shape: {x_raw.shape}')

kmeans = KMeans(n_clusters=n_clusters, verbose=0, random_state=42)
raw_labels = kmeans.fit_predict(x_raw)
print('Fitting done')

x, indices = np.unique(x_raw, return_index=True, axis=0)
labels = raw_labels[indices]
print(f'Unique data shape: {x.shape}')

centers = kmeans.cluster_centers_
for ind_cluster, center in tqdm(enumerate(centers)):
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    for row in axes:
        for ax in row:
            ax.set_axis_off()
    mask = (labels == ind_cluster)
    cur_x = x[mask]
    cur_indices = indices[mask]
    distances = ((cur_x - center) ** 2).sum(axis=1)
    indices_sorted = sorted(list(range(len(cur_x))), key=distances.__getitem__)
    n_display = min(n_rows * n_cols, len(cur_x))
    for i, display_index in enumerate(np.linspace(0, len(cur_x), num=n_display, endpoint=False, dtype=np.int)):
        ax = axes[i % n_rows, i // n_rows]
        s, s_prime = train_dataset.get_raw(cur_indices[indices_sorted[display_index]])
        grid_s = get_grid(s, pixels_per_tile=pixels_per_tile)
        grid_s_prime = get_grid(s_prime, pixels_per_tile=pixels_per_tile)
        grid = join_grids(grid_s, grid_s_prime, pixels_between=pixels_between)
        ax.imshow(grid)
        ax.set_title(f'Distance: {distances[indices_sorted[display_index]]:.0f}')
    fig.suptitle(f'Cluster {ind_cluster}. N_elements: {(raw_labels == ind_cluster).sum()}, n_unique: {len(cur_x)}')
    # plt.tight_layout()
    plt.savefig(fname=save_path / f'cluster_{ind_cluster}.png')

# TODO: прислать картинки (или в WandN)
