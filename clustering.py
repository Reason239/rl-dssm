from utils import DatasetFromPickle, my_collate_fn
from gridworld import join_grids, grid_from_state_data, stage_and_pos_from_state_data
from dssm import DSSM, DSSMEmbed

from collections import Counter, defaultdict
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pathlib
import pickle
from scipy.stats import entropy

dataset_path = pathlib.Path('datasets/all_100')
experiment_path_base = pathlib.Path('experiments')
experiment_name = 'reg'
clustering_name = 'clustering_10'
save_path = experiment_path_base / experiment_name / clustering_name
save_path.mkdir(parents=True, exist_ok=True)
batch_size = 256
embed_size = 64
n_clusters = 10
n_cols = 6
n_rows = 10
figsize = (14, 14)
pixels_per_tile = 10
pixels_between = pixels_per_tile // 2

train_dataset = DatasetFromPickle(dataset_path / 'train.pkl', 'bool', dataset_path / 'state_data_train.pkl')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                              collate_fn=my_collate_fn)
x_raw_path = save_path / 'x_raw.pkl'
kmeans_path = save_path / 'kmeans.pkl'
if x_raw_path.exists() and kmeans_path.exists():
    print('Loading precomputed clustering')
    with open(x_raw_path, 'rb') as f:
        x_raw = pickle.load(f)
    with open(kmeans_path, 'rb') as f:
        kmeans = pickle.load(f)
else:
    print('Clustering')

    model = DSSM(in_channels=7, height=5, width=5, embed_size=embed_size)
    model.eval()
    model.load_state_dict(torch.load(experiment_path_base / experiment_name / 'best_model.pth'))

    embeds = []
    with torch.no_grad():
        for (s, s_prime), (_, _) in tqdm(train_dataloader):
            if isinstance(model, DSSM):
                embeds.append(model.phi2(s_prime - s).numpy())
            else:
                embeds.append(model.phi2(model.embed(s_prime) - model.embed(s)).numpy())

    x_raw = np.concatenate(embeds)
    kmeans = KMeans(n_clusters=n_clusters, verbose=0, random_state=42)
    kmeans.fit(x_raw)
    print('Fitting done')
    with open(x_raw_path, 'wb') as f:
        pickle.dump(x_raw, f)
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans, f)

print(f'Initial data shape: {x_raw.shape}')
raw_labels = kmeans.predict(x_raw)

x, indices = np.unique(x_raw, return_index=True, axis=0)
labels = raw_labels[indices]
print(f'Unique data shape: {x.shape}')

centers = kmeans.cluster_centers_


def make_plots(dtype='bool', n_buttons=3):
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
        for i, display_index in enumerate(np.linspace(0, len(cur_x), num=n_display, endpoint=False, dtype=int)):
            ax = axes[i % n_rows, i // n_rows]
            s_data, s_prime_data = train_dataset.get_state_data(cur_indices[indices_sorted[display_index]])
            grid_s = grid_from_state_data(*s_data)
            grid_s_prime = grid_from_state_data(*s_prime_data)
            grid = join_grids(grid_s, grid_s_prime, pixels_between=pixels_between)
            ax.imshow(grid)
            ax.set_title(f'Distance: {distances[indices_sorted[display_index]]:.0f}')
        fig.suptitle(f'Cluster {ind_cluster}. n_elements: {(raw_labels == ind_cluster).sum()}, n_unique: {len(cur_x)}')
        # plt.tight_layout()
        plt.savefig(fname=save_path / f'cluster_{ind_cluster}.png')
        plt.cla()
        plt.clf()
        plt.close(fig)


colors = ['red', 'green', 'blue', 'orange', 'purple']


def compute_statistics(dtype='bool', n_buttons=3, grid_size=(5, 5)):
    info = [{} for c in centers]
    for ind_cluster, (center, info_dict) in tqdm(enumerate(zip(centers, info)), total=len(centers)):
        raw_mask = (raw_labels == ind_cluster)
        cur_x_raw = x_raw[raw_mask]
        cur_raw_indices = np.arange(len(x_raw))[raw_mask]
        n_elements = raw_mask.sum()
        matrix_stage = np.zeros((3 * n_buttons, 3 * n_buttons))
        matrix_pos_s = np.zeros(grid_size)
        matrix_pos_s_prime = np.zeros(grid_size)
        for ind in cur_raw_indices:
            s_data, s_prime_data = train_dataset.get_state_data(ind)
            stage_s, pos_s = stage_and_pos_from_state_data(*s_data)
            stage_s_prime, pos_s_prime = stage_and_pos_from_state_data(*s_prime_data)
            matrix_stage[stage_s, stage_s_prime] += 1
            matrix_pos_s[pos_s[0], pos_s[1]] += 1
            matrix_pos_s_prime[pos_s_prime[0], pos_s_prime[1]] += 1
        fig = plt.figure(constrained_layout=True, figsize=(10, 14))
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, :])
        ax1.set_title(f'Stages for cluster {ind_cluster}')
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
        plt.savefig(fname=save_path / f'heat_{ind_cluster}.png')
        plt.cla()
        plt.clf()
        plt.close(fig)

if __name__ == '__main__':
    # make_plots()
    # compute_statistics()
    print(x_raw.shape)
    x0 = x_raw[0]
    print((x0 ** 2).sum())
    x0 = x_raw[1]
    print((x0 ** 2).sum())
    print((x_raw[:40] ** 2).sum(axis=1))
