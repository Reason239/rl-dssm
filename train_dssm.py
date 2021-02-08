from utils import DatasetFromPickle, format_time
from dssm import DSSM

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import json
import pickle
from tqdm import tqdm
import pathlib
from timeit import default_timer as timer

time_start = timer()
np.random.seed(42)
dataset_path = 'datasets/all_1000/'
experiment_path_base = 'experiments/'
experiment_name = 'test_cpu'
# TODO try bigger batch_size
batch_size = 64
n_epochs = 3
patience = max(1, n_epochs // 5)
embed_size = 64
device = 'cpu'

experiment_path = pathlib.Path(experiment_path_base + experiment_name)
experiment_path.mkdir(parents=True, exist_ok=True)

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
best_test_acc = -1
best_epoch = -1
n_epochs_run = 0

model = model.to(device)

# training loop
for epoch in tqdm(range(n_epochs)):
    # train
    train_loss = 0
    train_correct = 0
    train_total = 0
    model.train()
    for s, s_prime in train_dataloader:
        s = s.to(device)
        s_prime = s_prime.to(device)
        output = model((s, s_prime))
        loss = criterion(output, target)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        train_loss += loss.cpu().item()
        _, predicted = torch.max(output.data, 1)
        train_total += batch_size
        train_correct += predicted.eq(target.data).cpu().sum().item()
    train_losses.append(train_loss / train_total)
    train_accs.append(train_correct / train_total)

    # eval
    test_loss = 0
    test_correct = 0
    test_total = 0
    model.eval()
    for s, s_prime in test_dataloader:
        s = s.to(device)
        s_prime = s_prime.to(device)
        output = model((s, s_prime))
        loss = criterion(output, target)

        test_loss += loss.cpu().item()
        _, predicted = torch.max(output.data, 1)
        test_total += batch_size
        test_correct += predicted.eq(target.data).cpu().sum().item()
    test_losses.append(test_loss / test_total)
    test_accs.append(test_correct / test_total)

    # save model (best)
    if test_accs[-1] > best_test_acc:
        best_test_acc = test_accs[-1]
        best_epoch = epoch
        torch.save(model.state_dict(), experiment_path / 'best_model.pth')

    # stop if test accuracy isn't going up
    if epoch > best_epoch + patience:
        n_epochs_run = epoch
        break
else:
    # if not break:
    n_epochs_run = n_epochs
time_end = timer()
time_str = format_time(time_end - time_start)

# save experiment data
metrics = {'train_losses': train_losses, 'test_losses': test_losses,
           'train_accs': train_accs, 'test_accs': test_accs}
with open(experiment_path / 'metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

info = {'dataset': dataset_path, 'batch_size': batch_size, 'n_epochs': n_epochs, 'n_epochs_run': n_epochs_run,
        'best_epoch': best_epoch, 'best_test_acc': best_test_acc, 'time_str': time_str, 'device': str(device)}
with open(experiment_path / 'info.json', 'w') as f:
    json.dump(info, f)

print('Done')
