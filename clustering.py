from utils import DatasetFromPickle
from dssm import DSSM

from operator import itemgetter
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

dataset_path = 'datasets/all_1000/'
experiment_path_base = 'experiments/'
experiment_name = 'test_bs64_l'
batch_size = 256
embed_size = 64

train_dataset = DatasetFromPickle(dataset_path + 'train.pkl')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

model = DSSM(in_channels=7, height=5, width=5, embed_size=embed_size)
model.eval()
model.load_state_dict(torch.load(experiment_path_base + experiment_name + '/best_model.pth'))

embeds = []
with torch.no_grad():
    for s, s_prime in tqdm(train_dataloader):
        embeds.append(model.phi2(s_prime - s).numpy())

x = np.concatenate(embeds)
print(f'Data shape: {x.shape}')

kmeans = KMeans(n_clusters=8, verbose=0, random_state=42)
kmeans.fit(x)

centers = kmeans.cluster_centers_
center = centers[0]
distances = ((x - center)**2).sum(axis=1)
ind_sorted = sorted(list(range(len(x))), key=distances.__getitem__)


print('Fitting done')


