import comet_ml
import torch
from train_dssm import train_dssm
from itertools import product

dataset_name = 'all_100'
experiment_name = 'reg_test'
use_comet = False
comet_tag = None
comet_disabled = False  # For debugging
save = False
n_trajectories = 16
pairs_per_trajectory = 4
n_epochs = 60
patience_ratio = 0.5
embed_size = 64
device = 'cpu'
do_eval = True
model_type = 'DSSM'
state_embed_size = 3
embed_conv_channels = None
n_z = 10
commitment_cost = 0.25  # strength of encoder penalty for distance from quantized embeds
dssm_embed_loss_coef = 1.
dssm_z_loss_coef = None
distance_loss_coef = 0.1  # coefficient of distance losses in total loss
dssm_eps = 1e-4  # epsilon for normalization of DSSM embeds
do_quantize = True  # effectively turns DSSMEmbed into DSSM with state embeddings

train_dssm(experiment_name, dataset_name, use_comet, comet_tag, comet_disabled, save, n_trajectories, pairs_per_trajectory,
           n_epochs, patience_ratio, embed_size, device, do_eval, model_type, state_embed_size, embed_conv_channels,
           n_z, commitment_cost, dssm_embed_loss_coef, dssm_z_loss_coef, distance_loss_coef, dssm_eps, do_quantize)

# comet_tag = 'coef_gs'
# dssm_embed_loss_coefs = [0.3, 1, 3]
# dssm_z_loss_coefs = [0.3, 1, 3]
# distance_loss_coefs = [0.03, 0.1, 0.3]
# for dssm_embed_loss_coef, dssm_z_loss_coef, distance_loss_coef in \
#         product(dssm_embed_loss_coefs, dssm_z_loss_coefs, distance_loss_coefs):
#     experiment_name = f'gs_q10_d{distance_loss_coef}_c025_z{dssm_z_loss_coef}_e{dssm_embed_loss_coef}'
#     print(experiment_name)
#     train_dssm(experiment_name, dataset_name, use_comet, comet_tag, comet_disabled, save, n_trajectories, pairs_per_trajectory,
#                n_epochs, patience_ratio, embed_size, device, do_eval, model_type, state_embed_size, embed_conv_channels,
#                n_z, commitment_cost, dssm_embed_loss_coef, dssm_z_loss_coef, distance_loss_coef, dssm_eps, do_quantize)
