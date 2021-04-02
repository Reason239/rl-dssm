from utils import DatasetFromPickle, format_time, BatchIterator, SmallBatchIterator
from dssm import DSSM, DSSMEmbed

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
dataset_path = 'datasets/int_1000/'
experiment_path_base = 'experiments/'
experiment_name = 'quant_q10_test3'
save = True
# TODO try bigger batch_size
# batch_size = 256
batches_per_update = 1
n_trajectories = 16
pairs_per_trajectory = 4
batch_size = n_trajectories * pairs_per_trajectory
n_epochs = 60
patience = max(1, n_epochs // 3)
embed_size = 64
device = 'cuda'
do_eval = True
if device == 'cuda':
    torch.cuda.empty_cache()
dtype_for_torch = 'int'
state_embed_size = 3
embed_conv_channels = None
n_z = 10
dssm_eps = 1e-4

if save:
    experiment_path = pathlib.Path(experiment_path_base + experiment_name)
    experiment_path.mkdir(parents=True, exist_ok=True)

# prepare datasets
# train_dataset = DatasetFromPickle(dataset_path + 'train.pkl')
# test_dataset = DatasetFromPickle(dataset_path + 'test.pkl')
#
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_batches = BatchIterator(dataset_path + 'train.pkl', dataset_path + 'idx_train.pkl',
                              n_trajectories, pairs_per_trajectory, None, dtype_for_torch)
test_batches = BatchIterator(dataset_path + 'test.pkl', dataset_path + 'idx_test.pkl',
                             n_trajectories, pairs_per_trajectory, None, dtype_for_torch)
# train_batches = SmallBatchIterator(dataset_path + 'train.pkl', dataset_path + 's_ind_train.pkl')
# test_batches = SmallBatchIterator(dataset_path + 'test.pkl', dataset_path + 's_ind_test.pkl')

# prepare the model
# model = DSSM(in_channels=7, height=5, width=5, embed_size=embed_size)
model = DSSMEmbed(dict_size=14, height=5, width=5, embed_size=embed_size, state_embed_size=state_embed_size,
                  embed_conv_channels=embed_conv_channels, n_z=n_z, eps=dssm_eps, commitment_cost=0.25)
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
tqdm_range = tqdm(range(n_epochs))
for epoch in tqdm_range:
    # train
    train_loss = 0
    train_correct = 0
    train_total = 0
    model.train()
    train_batches.refresh()
    loss = 0
    for ind, (s, s_prime) in enumerate(train_batches):
        s = s.to(device)
        s_prime = s_prime.to(device)
        output, batch_loss = model.forward_and_loss((s, s_prime), criterion, target)
        # target = torch.arange(0, len(s)).to(device)
        # loss += criterion(output, target)
        loss += batch_loss

        if (ind + 1) % batches_per_update == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loss = 0

        _, predicted = torch.max(output.data, 1)
        train_total += len(s)
        train_correct += predicted.eq(target.data).sum().item()
    train_losses.append(train_loss / train_total)
    train_accs.append(train_correct / train_total)

    # eval
    if do_eval:
        test_loss = 0
        test_correct = 0
        test_total = 0
        model.eval()
        test_batches.refresh()
        for s, s_prime in test_batches:
            s = s.to(device)
            s_prime = s_prime.to(device)
            output, batch_loss = model.forward_and_loss((s, s_prime), criterion, target)
            # target = torch.arange(0, len(s)).to(device)
            # loss = criterion(output, target)

            test_loss += batch_loss.item()
            _, predicted = torch.max(output.data, 1)
            test_total += len(s)
            test_correct += predicted.eq(target.data).sum().item()
        test_losses.append(test_loss / test_total)
        test_accs.append(test_correct / test_total)

        # save model (best)
        if test_accs[-1] > best_test_acc:
            best_test_acc = test_accs[-1]
            best_epoch = epoch
            if save:
                torch.save(model.state_dict(), experiment_path / 'best_model.pth')
        tqdm_range.set_postfix(train_loss=train_losses[-1], test_loss=test_losses[-1], test_acc=test_accs[-1])

        # stop if test accuracy isn't going up
        if epoch > best_epoch + patience:
            n_epochs_run = epoch
            break
    else:
        if save:
            torch.save(model.state_dict(), experiment_path / 'best_model.pth')


else:
    # if not break:
    n_epochs_run = n_epochs
time_end = timer()
time_str = format_time(time_end - time_start)

# save experiment data
if save:
    metrics = {'train_losses': train_losses, 'test_losses': test_losses,
               'train_accs': train_accs, 'test_accs': test_accs}
    with open(experiment_path / 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    info = {'dataset': dataset_path, 'batch_size': batch_size, 'n_trajectories': n_trajectories,
            'pairs_per_trajectory': pairs_per_trajectory, 'n_epochs': n_epochs, 'n_epochs_run': n_epochs_run,
            'best_epoch': best_epoch, 'best_test_acc': best_test_acc, 'time_str': time_str, 'device': str(device),
            'embed_size': embed_size, 'dtype_for_torch': dtype_for_torch, 'state_embed_size': state_embed_size,
            'n_z': n_z, 'dssm_eps': dssm_eps}
    with open(experiment_path / 'info.json', 'w') as f:
        json.dump(info, f, indent=4)

    with open(experiment_path_base + 'summary.csv', 'a') as f:
        f.write(experiment_name + f',{best_test_acc}')
