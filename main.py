import comet_ml
import torch
from train_dssm import train_dssm
from dssm import DSSM, DSSMEmbed, DSSMReverse
from utils import get_parameters_list

dataset_name = 'all_1000'
evaluate_dataset_name = 'evaluate_bool_1024_n4'
n_negatives = 4
evaluate_batch_size = 1
experiment_name = 'eval_reg_tr132'
model_type = 'DSSM'
use_comet = True
comet_tags = [model_type]
comet_disabled = False  # For debugging
save = True
n_trajectories = 1
pairs_per_trajectory = 32
n_epochs = 60
patience_ratio = 1.
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
n_z = 10
commitment_cost = 0.25  # strength of encoder penalty for distance from quantized embeds
dssm_embed_loss_coef = 1.
dssm_z_loss_coef = None
distance_loss_coef = 0.1  # coefficient of distance losses in total loss
dssm_eps = 1e-4  # epsilon for normalization of DSSM embeds
do_quantize = True  # effectively turns DSSMEmbed into DSSM with state embeddings

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
                                 embed_conv_channels=None, phi_conv_channels=phi_conv_channels, fc_sizes=fc_sizes,
                                 do_normalize=do_normalize, n_z=n_z, eps=dssm_eps, commitment_cost=commitment_cost,
                                 distance_loss_coef=distance_loss_coef, dssm_embed_loss_coef=dssm_embed_loss_coef,
                                 dssm_z_loss_coef=dssm_z_loss_coef, do_quantize=do_quantize)
    model = DSSMReverse(**base_model_parameters)
else:
    raise Exception(f'Incorrect model_type {model_type}, should be "DSSM", "DSSMEmbed" of "DSSMReverse"')

base_parameters = dict(model=model, experiment_name=experiment_name, evaluate_dataset_name=evaluate_dataset_name,
                       n_negatives=n_negatives, evaluate_batch_size=evaluate_batch_size, dataset_name=dataset_name,
                       use_comet=use_comet, comet_tags=comet_tags, comet_disabled=comet_disabled, save=save,
                       n_trajectories=n_trajectories, pairs_per_trajectory=pairs_per_trajectory, n_epochs=n_epochs,
                       patience_ratio=patience_ratio, device=device, do_eval=do_eval, do_quantize=do_quantize)
parameters_to_vary = {}
model_parameters_to_vary = {'embed_conv_size': [None, 3],
                            'dssm_z_loss_coef': [None, 1.],
                            'n_z': [10, 50]}
parameters_mode = None

if parameters_mode is None:
    print('Running a single experiment')
    train_dssm(model_kwargs=base_model_parameters, **base_parameters)
else:
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
