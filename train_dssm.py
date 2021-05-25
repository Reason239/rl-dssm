import comet_ml

from utils import DatasetFromPickle, BatchIterator, ValidationBatchIterator
from dssm import DSSM, DSSMEmbed, DSSMReverse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import json
import pickle
from tqdm import tqdm
import pathlib
from collections import defaultdict


def run_model(model, optimizer, batches, device, criterion, target, mode, do_quantize, downscale_factor=1):
    assert mode in ['train', 'validate', 'evaluate']
    losses = defaultdict(float)
    n_correct = 0
    n_total = 0
    z_inds_count = 0
    if mode == 'train':
        model.train()
    else:
        model.eval()
    batches.refresh()
    for ind, (s, s_prime) in enumerate(batches):
        # get the inputs
        s = s.to(device)
        s_prime = s_prime.to(device)

        if mode == 'train':
            # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
        results = model.forward_and_loss((s, s_prime), criterion, target, downscale_factor)
        output = results['output']
        loss = results['total_loss']
        if mode == 'train':
            loss.backward()
            optimizer.step()

        for key, value in results.items():
            if 'loss' in key:
                losses[key] += value.item()
        _, predicted = torch.max(output.data, 1)
        n_total += len(predicted)
        n_correct += predicted.eq(target.data).sum().item()

        # Log quantized vectors indices count
        if (isinstance(model, DSSMEmbed) or isinstance(model, DSSMReverse)) and do_quantize:
            z_inds_count += results['z_inds_count']

    for key in losses:
        losses[key] /= len(batches)
    acc = n_correct / n_total

    results = dict(losses)
    results['accuracy'] = acc
    if (isinstance(model, DSSMEmbed) or isinstance(model, DSSMReverse)) and do_quantize:
        results['z_inds_count'] = z_inds_count

    return results


def train_dssm(model, experiment_name, dataset_name='int_1000', evaluate_dataset_name='evaluate_1024_n4', n_negatives=4,
               evaluate_batch_size=64, use_comet=False, comet_tags=None, comet_disabled=False, save=False,
               n_trajectories=16, pairs_per_trajectory=4, n_epochs=60, patience_ratio=1, device='cpu', do_eval=True,
               do_quantize=True, model_kwargs=None, train_on_synthetic_dataset=False):
    np.random.seed(42)
    dataset_path = f'datasets/{dataset_name}/'
    evaluate_dataset_path = f'datasets/{evaluate_dataset_name}/'
    experiment_path_base = 'experiments/'
    batch_size = n_trajectories * pairs_per_trajectory
    patience = max(1, int(n_epochs * patience_ratio))
    if device == 'cuda':
        torch.cuda.empty_cache()
    dtype_for_torch = np.float32 if isinstance(model, DSSM) else int  # int for embeddings, float for convolutions

    # Setup Comet.ml
    if use_comet:
        comet_experiment = comet_ml.Experiment(project_name='gridworld_dssm', auto_metric_logging=False,
                                               disabled=comet_disabled)
        comet_experiment.set_name(experiment_name)
        comet_experiment.log_parameters(model_kwargs)
        if comet_tags:
            comet_experiment.add_tags(comet_tags)
    else:
        comet_experiment = None

    # Setup local save path
    if save:
        experiment_path = pathlib.Path(experiment_path_base + experiment_name)
        experiment_path.mkdir(parents=True, exist_ok=True)

    # Prepare datasets
    # train_dataset = DatasetFromPickle(dataset_path + 'train.pkl')
    # test_dataset = DatasetFromPickle(dataset_path + 'test.pkl')
    #
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if not train_on_synthetic_dataset:
        train_batches = BatchIterator(dataset_path + 'train.pkl', dataset_path + 'idx_train.pkl',
                                      n_trajectories, pairs_per_trajectory, None, dtype_for_torch)
        test_batches = BatchIterator(dataset_path + 'test.pkl', dataset_path + 'idx_test.pkl',
                                     n_trajectories, pairs_per_trajectory, None, dtype_for_torch)
    else:
        train_batches = ValidationBatchIterator(dataset_path + 'train.pkl', dataset_path + 'idx_train.pkl',
                                                n_negatives, evaluate_batch_size, dtype_for_torch)
        test_batches = ValidationBatchIterator(dataset_path + 'test.pkl', dataset_path + 'idx_test.pkl',
                                               n_negatives, evaluate_batch_size, dtype_for_torch)
    evaluate_batches = ValidationBatchIterator(evaluate_dataset_path + 'evaluate.pkl',
                                               evaluate_dataset_path + 'idx_evaluate.pkl', n_negatives,
                                               evaluate_batch_size, dtype_for_torch)

    criterion = nn.CrossEntropyLoss()
    downscale_factor = n_negatives + 1
    evaluate_target = torch.zeros(evaluate_batch_size, dtype=torch.long).to(device)
    if not train_on_synthetic_dataset:
        target = torch.arange(0, batch_size).to(device)
        train_downscale_factor = 1
    else:
        target = evaluate_target
        train_downscale_factor = downscale_factor

    optimizer = torch.optim.Adam(model.parameters())

    test_accs = []
    best_test_acc = -1
    best_epoch = -1

    model = model.to(device)

    # training loop
    tqdm_range = tqdm(range(n_epochs))
    for epoch in tqdm_range:
        # train
        mode = 'train'
        results = run_model(model, optimizer, train_batches, device, criterion, target, mode, do_quantize,
                            train_downscale_factor)
        # Log all metrics
        if use_comet:
            for key, value in results.items():
                if key != 'z_inds_count':
                    comet_experiment.log_metric(f'{mode}_{key}', value)
            comet_experiment.log_metric('dssm_scale', torch.exp(model.scale).item())
        if (isinstance(model, DSSMEmbed) or isinstance(model, DSSMReverse)) and do_quantize:
            z_inds_count = results['z_inds_count']
            z_inds_count = z_inds_count / z_inds_count.sum()
            comet_experiment.log_text('Counts: ' + ' '.join(f'{num:.1%}' for num in z_inds_count))

        if do_eval:
            # Validate
            mode = 'validate'
            results = run_model(model, optimizer, test_batches, device, criterion, target, mode, do_quantize,
                                train_downscale_factor)
            # Log all metrics
            if use_comet:
                for key, value in results.items():
                    if key != 'z_inds_count':
                        comet_experiment.log_metric(f'{mode}_{key}', value)
            # Log embedding distance matrix
            if isinstance(model, DSSMEmbed) or isinstance(model, DSSMReverse):
                z_vectors = model.z_vectors_norm if isinstance(model, DSSMEmbed) else model.get_z_vectors()
                z_vectors = z_vectors.detach()
                z_vectors_batch = z_vectors.unsqueeze(0)
                embed_dist_matr = torch.cdist(z_vectors_batch, z_vectors_batch).squeeze().cpu().numpy()
                np.fill_diagonal(embed_dist_matr, torch.sqrt((z_vectors ** 2).sum(axis=1)).cpu().numpy())
                comet_experiment.log_confusion_matrix(matrix=embed_dist_matr, title='Embeddings distance matrix')
            test_accs.append(results['accuracy'])

            # Evaluate
            mode = 'evaluate'
            results = run_model(model, optimizer, evaluate_batches, device, criterion, evaluate_target, mode,
                                do_quantize, downscale_factor)
            # Log all metrics
            if use_comet:
                for key, value in results.items():
                    if key != 'z_inds_count':
                        comet_experiment.log_metric(f'{mode}_{key}', value)

            # save model (best)
            if test_accs[-1] > best_test_acc:
                best_test_acc = test_accs[-1]
                best_epoch = epoch
                if save:
                    torch.save(model.state_dict(), experiment_path / 'best_model.pth')
            tqdm_range.set_postfix(test_acc=test_accs[-1], eval_acc=results['accuracy'])

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
                    dtype_for_torch=dtype_for_torch)

        if save:
            with open(experiment_path / 'info.json', 'w') as f:
                json.dump(info.update(model_kwargs), f, indent=4)

            with open(experiment_path / 'model_kwargs.pkl', 'wb') as f:
                pickle.dump(model_kwargs, f)

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
