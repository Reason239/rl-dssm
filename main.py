"""The main script that is executed for the model training experiments. No command-line options support yet"""

import comet_ml
import torch

from train_dssm import train_dssm
from dssm import DSSM, DSSMEmbed, DSSMReverse
from utils import get_parameters_list

# Parameters configuration (refer to train_dssm.train_dssm() function documentation)
dataset_name = 'all_1000'
# dataset_name = 'synth_int_20480_n4'
evaluate_dataset_name = 'evaluate_bool_1024_n4'
n_negatives = 4
evaluate_batch_size = 64
experiment_name = 'reg_test'
model_type = 'DSSM'
use_comet = True
comet_tags = [model_type]
comet_disabled = False  # For debugging
save = True
n_trajectories = 16
pairs_per_trajectory = 4
n_epochs = 60
patience_ratio = 1.
embed_size = 64
device = 'cpu'
do_eval = True
# for DSSMEmbed model
embed_conv_size = None
# for DSSMReverse model (in development)
embed_conv_channels = [3, 3]
phi_conv_channels = [16, 32]
fc_sizes = [embed_size, embed_size]
do_normalize = False
# For both DSSMEmbed and DSSMReverse models
state_embed_size = 3
n_z = 10
commitment_cost = 0.25  # strength of encoder penalty for distance from quantized embeds
dssm_embed_loss_coef = 1.
dssm_z_loss_coef = None
distance_loss_coef = 0.1  # coefficient of distance losses in total loss
dssm_eps = 1e-4  # epsilon for normalization of DSSM embeds
do_quantize = True  # effectively turns DSSMEmbed into DSSM with state embeddings

# Automatic config
train_on_synthetic_dataset = 'synth' in dataset_name
if train_on_synthetic_dataset:
    comet_tags += ['synth']

# Make a model instance, remember the parameters
if model_type == 'DSSM':
    base_model_parameters = dict(in_channels=7, height=5, width=5, embed_size=embed_size)
    model = DSSM(**base_model_parameters)
elif model_type == 'DSSMEmbed':
    base_model_parameters = dict(dict_size=14, height=5, width=5, embed_size=embed_size,
                                 state_embed_size=state_embed_size,
                                 embed_conv_size=embed_conv_size, n_z=n_z, eps=dssm_eps,
                                 commitment_cost=commitment_cost,
                                 distance_loss_coef=distance_loss_coef, dssm_embed_loss_coef=dssm_embed_loss_coef,
                                 dssm_z_loss_coef=dssm_z_loss_coef, do_quantize=do_quantize)
    model = DSSMEmbed(**base_model_parameters)
elif model_type == 'DSSMReverse':
    base_model_parameters = dict(dict_size=14, height=5, width=5, embed_size=embed_size,
                                 state_embed_size=state_embed_size,
                                 embed_conv_channels=embed_conv_channels, phi_conv_channels=phi_conv_channels,
                                 fc_sizes=fc_sizes,
                                 do_normalize=do_normalize, n_z=n_z, eps=dssm_eps, commitment_cost=commitment_cost,
                                 distance_loss_coef=distance_loss_coef, dssm_embed_loss_coef=dssm_embed_loss_coef,
                                 dssm_z_loss_coef=dssm_z_loss_coef, do_quantize=do_quantize)
    model = DSSMReverse(**base_model_parameters)
else:
    raise Exception(f'Incorrect model_type {model_type}, should be "DSSM", "DSSMEmbed" of "DSSMReverse"')

# Remember the experiment parameters specified in the config
base_parameters = dict(model=model, experiment_name=experiment_name, evaluate_dataset_name=evaluate_dataset_name,
                       n_negatives=n_negatives, evaluate_batch_size=evaluate_batch_size, dataset_name=dataset_name,
                       use_comet=use_comet, comet_tags=comet_tags, comet_disabled=comet_disabled, save=save,
                       n_trajectories=n_trajectories, pairs_per_trajectory=pairs_per_trajectory, n_epochs=n_epochs,
                       patience_ratio=patience_ratio, device=device, do_eval=do_eval, do_quantize=do_quantize,
                       train_on_synthetic_dataset=train_on_synthetic_dataset)

# Config for a grid search or a series of experiments
parameters_mode = None  # None, 'gs' or 'series'
parameters_to_vary = {}
model_parameters_to_vary = {'embed_conv_size': [None, 3],
                            'dssm_z_loss_coef': [None, 1.],
                            'n_z': [10, 50]}

if parameters_mode is None:
    print('Running a single experiment')
    train_dssm(model_kwargs=base_model_parameters, **base_parameters)
else:
    # Run several experiments
    all_parameters_list = get_parameters_list(base_parameters, parameters_to_vary,
                                              base_model_parameters, model_parameters_to_vary, mode=parameters_mode)
    total = len(all_parameters_list)
    for i, params in enumerate(all_parameters_list):
        print(f'\nRunning experiment {i + 1}/{total}\n')
        if model_type == 'DSSM':
            model = DSSM(**params['model_kwargs'])
        elif model_type == 'DSSMEmbed':
            model = DSSMEmbed(**params['model_kwargs'])
        else:
            model = DSSMReverse(**params['model_kwargs'])
        train_dssm(**params)
