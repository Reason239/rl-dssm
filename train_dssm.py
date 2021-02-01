from utils import DatasetFromPickle
from dssm import DSSM

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import json
import pickle
from tqdm import tqdm

np.random.seed(42)
dataset_path = 'datasets/all_1000/'
experiment_path = 'experiments/'
experiment_name = 'test_wtf'
# TODO try bigger batch_size
batch_size = 64
n_epochs = 1
embed_size = 64
device = 'cpu'

# prepare datasets
train_dataset = DatasetFromPickle(dataset_path + 'train.pkl')
test_dataset = DatasetFromPickle(dataset_path + 'test.pkl')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# prepare the model
model = DSSM(in_channels=7, height=5, width=5, embed_size=embed_size)
criterion = nn.CrossEntropyLoss()
target = torch.arange(0, batch_size).to(device)
optimizer = torch.optim.Adam(model.parameters())

train_losses = []
test_losses = []
train_accs = []
test_accs = []

model.to(device)

# training loop
for epoch in tqdm(range(n_epochs)):
    # train
    train_loss = 0
    train_correct = 0
    train_total = 0
    model.train()
    for s, s_prime in train_dataloader:
        s.to(device)
        s_prime.to(device)
        output = model((s, s_prime))
        loss = criterion(output, target)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        train_loss += loss.cpu().item()
        _, predicted = torch.max(output.data, 1)
        train_total += batch_size
        train_correct += predicted.eq(target.data).cpu().sum()
    train_losses.append(train_loss / train_total)
    train_accs.append(train_correct / train_total)

    # eval
    test_loss = 0
    test_correct = 0
    test_total = 0
    model.eval()
    for s, s_prime in test_dataloader:
        s.to(device)
        s_prime.to(device)
        output = model((s, s_prime))
        loss = criterion(output, target)

        test_loss += loss.cpu().item()
        _, predicted = torch.max(output.data, 1)
        test_total += batch_size
        test_correct += predicted.eq(target.data).cpu().sum()
    test_losses.append(test_loss / test_total)
    test_accs.append(test_correct / test_total)

# save experiment data
metrics = {'train_losses': train_losses, 'test_losses': test_losses,
           'train_accs': train_accs, 'test_accs': test_accs}
with open(experiment_path + experiment_name + '/metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

params = {'dataset': dataset_path, 'batch_size': batch_size, 'n_epochs': n_epochs}
with open(experiment_path + experiment_name + '/parameters.json', 'w') as f:
    json.dump(params, f)

print('Done')
