import comet_ml
import torch
from train_dssm import train_dssm
from dssm import DSSM, DSSMEmbed, DSSMReverse
from itertools import product
from utils import get_kwargs_grid_list

dataset_name = 'int_1000'
experiment_name = 'quant_q50_2_32'
model_type = 'DSSMEmbed'
use_comet = True
comet_tags = [model_type]
comet_disabled = False  # For debugging
save = True
n_trajectories = 2
pairs_per_trajectory = 32
n_epochs = 60
patience_ratio = 0.5
embed_size = 64
device = 'cpu'
do_eval = True
# DSSMEmbed
embed_conv_size = None
# DSSMReverse
embed_conv_channels = [3, 3]
phi_conv_channels = [16, 32]
fc_sizes = [embed_size, embed_size]
do_normalize = False
# DSSMEmbed and DSSMReverse
state_embed_size = 3
n_z = 50
commitment_cost = 0.25  # strength of encoder penalty for distance from quantized embeds
dssm_embed_loss_coef = 1.
dssm_z_loss_coef = None
distance_loss_coef = 0.1  # coefficient of distance losses in total loss
dssm_eps = 1e-4  # epsilon for normalization of DSSM embeds
do_quantize = True  # effectively turns DSSMEmbed into DSSM with state embeddings

if model_type == 'DSSM':
    kwargs = dict(in_channels=7, height=5, width=5, embed_size=embed_size)
    model = DSSM(**kwargs)
elif model_type == 'DSSMEmbed':
    kwargs = dict(dict_size=14, height=5, width=5, embed_size=embed_size, state_embed_size=state_embed_size,
                  embed_conv_size=embed_conv_size, n_z=n_z, eps=dssm_eps,
                  commitment_cost=commitment_cost,
                  distance_loss_coef=distance_loss_coef, dssm_embed_loss_coef=dssm_embed_loss_coef,
                  dssm_z_loss_coef=dssm_z_loss_coef, do_quantize=do_quantize)
    model = DSSMEmbed(**kwargs)
elif model_type == 'DSSMReverse':
    kwargs = dict(dict_size=14, height=5, width=5, embed_size=embed_size, state_embed_size=state_embed_size,
                  embed_conv_channels=None, phi_conv_channels=phi_conv_channels, fc_sizes=fc_sizes,
                  do_normalize=do_normalize, n_z=n_z, eps=dssm_eps, commitment_cost=commitment_cost,
                  distance_loss_coef=distance_loss_coef, dssm_embed_loss_coef=dssm_embed_loss_coef,
                  dssm_z_loss_coef=dssm_z_loss_coef, do_quantize=do_quantize)
    model = DSSMReverse(**kwargs)
else:
    raise Exception(f'Incorrect model_type {model_type}, should be "DSSM" or "DSSMEmbed"')

# parameters_grid = {'fc_sizes': [None, [embed_size, embed_size]],
#                    'do_normalize': [False, True],
#                    'dssm_z_loss_coef': [None, 1.],
#                    'n_z': [10, 50]}
parameters_grid = None

if parameters_grid is None:
    train_dssm(model=model, experiment_name=experiment_name, dataset_name=dataset_name, use_comet=use_comet,
               comet_tags=comet_tags, comet_disabled=comet_disabled, save=save, n_trajectories=n_trajectories,
               pairs_per_trajectory=pairs_per_trajectory, n_epochs=n_epochs, patience_ratio=patience_ratio,
               device=device,
               do_eval=do_eval, do_quantize=do_quantize, model_kwargs=kwargs)
else:
    comet_tags.append('gs')
    kwargs_grid_list = get_kwargs_grid_list(parameters_grid, kwargs)
    total = len(kwargs_grid_list)
    for i, gs_model_kwargs in enumerate(kwargs_grid_list):
        experiment_name = f'gs_{experiment_name}__{i + 1:02d}'
        print(f'\nRunning experiment {i + 1}/{total}\n')
        train_dssm(model=model, experiment_name=experiment_name, dataset_name=dataset_name, use_comet=use_comet,
                   comet_tags=comet_tags, comet_disabled=comet_disabled, save=save, n_trajectories=n_trajectories,
                   pairs_per_trajectory=pairs_per_trajectory, n_epochs=n_epochs, patience_ratio=patience_ratio,
                   device=device, do_eval=do_eval, do_quantize=do_quantize, model_kwargs=gs_model_kwargs)
