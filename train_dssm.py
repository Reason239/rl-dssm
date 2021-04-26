import comet_ml

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
from collections import defaultdict
from contextlib import nullcontext

np.random.seed(42)
dataset_path = 'datasets/int_1000/'
experiment_path_base = 'experiments/'
experiment_name = 'quant_q10_test4'
use_comet = True
save = True
# TODO try bigger batch_size
# batch_size = 256
n_trajectories = 16
pairs_per_trajectory = 4
batch_size = n_trajectories * pairs_per_trajectory
n_epochs = 60
patience = max(1, n_epochs // 3)
embed_size = 64
device = 'cpu'
do_eval = True
if device == 'cuda':
    torch.cuda.empty_cache()
dtype_for_torch = 'int'
state_embed_size = 3
embed_conv_channels = None
n_z = 50
dssm_eps = 1e-4

if use_comet:
    comet_experiment = comet_ml.Experiment(project_name='gridworld_dssm', auto_metric_logging=False)
    comet_experiment.set_name(experiment_name)
    train_context = comet_experiment.train
    eval_context = comet_experiment.validate
else:
    comet_experiment = None
    train_context = nullcontext
    eval_context = nullcontext

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
    with train_context():
        train_loss = 0
        train_correct = 0
        train_total = 0
        model.train()
        train_batches.refresh()
        for ind, (s, s_prime) in enumerate(train_batches):
            # get the inputs
            s = s.to(device)
            s_prime = s_prime.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output, loss = model.forward_and_loss((s, s_prime), criterion, target)
            # target = torch.arange(0, len(s)).to(device)
            # loss += criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            train_loss += loss_item

            _, predicted = torch.max(output.data, 1)
            train_total += len(s)
            train_correct += predicted.eq(target.data).sum().item()

            if use_comet:
                comet_experiment.log_metric('batch_loss', loss_item)
        train_loss /= train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        if use_comet:
            comet_experiment.log_metrics({'loss': train_loss, 'accuracy': train_acc})

    # eval
    if do_eval:
        with eval_context():
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
            test_loss /= test_total
            test_acc = test_correct / test_total
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            if use_comet:
                comet_experiment.log_metrics({'loss': test_loss, 'accuracy': test_acc})

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

# save experiment data
if save or use_comet:
    info = dict(dataset_path=dataset_path, batch_size=batch_size, n_trajectories=n_trajectories,
                pairs_per_trajectory=pairs_per_trajectory, n_epochs=n_epochs, n_epochs_run=n_epochs_run,
                best_epoch=best_epoch, best_test_acc=best_test_acc, device=str(device),
                embed_size=embed_size, dtype_for_torch=dtype_for_torch, state_embed_size=state_embed_size,
                n_z=n_z, dssm_eps=dssm_eps)

    if save:
        metrics = {'train_losses': train_losses, 'test_losses': test_losses,
                   'train_accs': train_accs, 'test_accs': test_accs}
        with open(experiment_path / 'metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        with open(experiment_path / 'info.json', 'w') as f:
            json.dump(info, f, indent=4)

        with open(experiment_path_base + 'summary.csv', 'a') as f:
            f.write(experiment_name + f',{best_test_acc}')

    if use_comet:
        comet_experiment.log_parameters(info)
        # save experiment key for later logging
        experiment_keys_path = pathlib.Path('experiments/comet_keys.pkl')
        if not experiment_keys_path.exists():
            experiment_keys = defaultdict(list)
        else:
            with open(experiment_keys_path, 'rb') as f:
                experiment_keys = pickle.load(f)
        experiment_keys[experiment_name].append(comet_experiment.get_key())
        with open(experiment_keys_path, 'wb') as f:
            pickle.dump(experiment_keys, f)