"""Here the DSSM models are implemented"""

import torch
from torch import nn


def calculate_inner_product_matrix(vectors1, vectors2, downscale_factor=1):
    """Calculates a matrix, containing inner products of two sets of vectors.

    If downscale_factor = 1, then it is a regular matrix multiplication of vectors1 and vectors2.T
    If downscale_factor = k > 1, then takes every k-th vector from vectors1 and takes its products only with
    a corresponding block of k vectors from vactors2. This is needed for training on a synthetic dataset, where we want
    to compare a positive only to its corresponding negatives and not to others: downscale_factor = n_negatives + 1

    :param vectors1: torch tensor of shape (batch_size * downscale_factor, embedding_size)
    :param vectors2: torch tensor of shape (batch_size * downscale_factor, embedding_size)
    :param downscale_factor: 1 or a divisor of vectors.shape[0]
    :return: torch tensor of shape (batch_size, batch_size) for downscale_factor = 1 or
        shape (batch_size, downscale_factor) for downscale_factor > 1
    """
    if downscale_factor == 1:
        return torch.matmul(vectors1, torch.transpose(vectors2, 0, 1))
    assert vectors1.shape[0] % downscale_factor == 0
    bs = vectors1.shape[0] // downscale_factor
    embedding_size = vectors1.shape[1]
    vectors1 = vectors1[torch.arange(0, bs * downscale_factor, downscale_factor)].view(bs, 1, embedding_size)
    vectors2 = torch.transpose(vectors2.view(bs, downscale_factor, embedding_size), 1, 2)
    return torch.bmm(vectors1, vectors2).view(bs, downscale_factor)


class DSSM(nn.Module):
    """
    'Basic' DSSM model.

    Accepts a batch of (s, s') pairs and calculates the matrix for the DSSM loss.
    """

    def __init__(self, in_channels=7, height=5, width=5, embed_size=64):
        """'Basic' model accepts observations from the 'bool' format in torch.float32 dtype and shape
        (batch_size, height, width, in_channels)

        :param in_channels: number of channels in the input observations. Equal to 2 * num_buttons + 1
        :param height: height of the environment gridworld (of the input 'image')
        :param width: width of the environment gridworld (of the input 'image')
        :param embed_size: dimension of the generated embeddings of s and s' (before inner product)
        """
        super().__init__()
        self.relu = nn.ReLU()

        # phi_1
        # batch * height * width * in_channels
        self.phi1_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        # batch * height * width * 16
        self.phi1_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # batch * height * width * 32 -> flatten
        self.phi1_linear = nn.Linear(height * width * 32, embed_size)
        self.phi1_norm = nn.LayerNorm(normalized_shape=embed_size)

        # phi_2 - same architecture as phi_1
        self.phi2_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.phi2_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.phi2_linear = nn.Linear(height * width * 32, embed_size)
        self.phi2_norm = nn.LayerNorm(normalized_shape=embed_size)

        # alpha (temperature scale)
        self.scale = torch.nn.Parameter(torch.ones((1,)))

    def phi1(self, s):
        """The first DSSM 'tower'. Generates an embedding of a state

        :param s: state observation (or representation)
        :return: embedding
        """
        x1 = self.relu(self.phi1_conv1(s))
        x1 = self.relu(self.phi1_conv2(x1))
        x1 = torch.flatten(x1, start_dim=1)
        embed1 = self.phi1_norm(self.phi1_linear(x1))
        return embed1

    def phi2(self, diff):
        """The second DSSM 'tower'. Generates an embedding of a state transition.

        :param diff: state transition (difference of two state representations)
        :return: embedding
        """
        x2 = self.relu(self.phi2_conv1(diff))
        x2 = self.relu(self.phi2_conv2(x2))
        x2 = torch.flatten(x2, start_dim=1)
        embed2 = self.phi2_norm(self.phi2_linear(x2))
        return embed2

    def forward(self, x, downscale_factor=1):
        """

        :param x: tuple of s and s' batches of some batch_size
        :param downscale_factor: should be equal to n_negatives + 1 for 'synthetic' datasets
        :return: torch tensor of shape (batch_size, batch_size) when downscale_factor = 1 or
            (synthetic batch size, downscale_factor) for downscale_factor > 1 - pairwise fitness scores
            (refer to calculate_inner_product_matrix function)
        """
        s, s_prime = x

        embed1 = self.phi1(s)
        embed2 = self.phi2(s_prime - s)

        # calculate inner products (usually it's Gramm matrix)
        inner_product_matrix = calculate_inner_product_matrix(embed1, embed2, downscale_factor)
        output = self.scale * inner_product_matrix

        return output

    def forward_and_loss(self, x, criterion, target, downscale_factor=1):
        """Calculates both output of the model and the loss.

        :param x: tuple of s and s' batches of some batch_size
        :param criterion: loss function
        :param target: target for the loss function
        :param downscale_factor: should be equal to n_negatives + 1 for 'synthetic' datasets
        :return: dict with output and losses (here only FPS-loss)
        """
        output = self.forward(x, downscale_factor)
        total_loss = criterion(output, target)
        results = dict(output=output, total_loss=total_loss, dssm_loss=total_loss)
        return results


def normalized(matrix, eps=0.):
    """L_2 normalises every row of a matrix of vectors. Adds eps to the denominator for stability"""
    return matrix / (torch.sqrt(torch.sum(matrix ** 2, dim=1, keepdim=True)) + eps)


class DSSMEmbed(nn.Module):
    """
    DSSM model with quantization in the second 'tower'

    Accepts a batch of (s, s') pairs and calculates the matrix for the DSSM loss.
    """

    def __init__(self, dict_size=14, height=5, width=5, embed_size=64, state_embed_size=3, embed_conv_size=None,
                 n_z=5, eps=1e-4, commitment_cost=0.25, distance_loss_coef=1., dssm_embed_loss_coef=1.,
                 dssm_z_loss_coef=None, do_quantize=True):
        """'Quantised' model accepts observations from the 'int' format in torch.int dtype and shape
        (batch_size, height, width)

        :param dict_size: size of the embedding dictionary. All of the values in the input should be in this range.
            Equal to 2 * (2 * n_buttons + 1)
        :param height: height of the environment gridworld (of the input 'image')
        :param width: width of the environment gridworld (of the input 'image')
        :param embed_size: dimension of the generated embeddings of s and s' (before inner product)
        :param state_embed_size: dimension of an embedding vector for each of the possible dict_size linds of pixel s in
            the image
        :param embed_conv_size: None or number of channels for an additional (shared )conv layer in the beginning of
            both towers for a more complex state representations
        :param n_z: number of the quantization vectors
        :param eps: epsilon for the normalization before the inner product calculation
        :param commitment_cost: coefficient of encoder_latent_loss that penalises generated embeddings for being far
            from the quantization vectors (refer to 'Neural Discrete Representation Learning' paper code:
            https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py)
        :param distance_loss_coef: coefficient for the 'quantization' loss in the total loss. Penalises generated
            embeddings and quantization vectors for being from each other
        :param dssm_embed_loss_coef: coefficient of the FPS-loss. Gradients are transferred from the quantised vectors
            to the ambeddings, generated by the second tower
        :param dssm_z_loss_coef: None or coefficient of the FPS-loss. Gradients are left on the the quantised vectors
        :param do_quantize: bool, whether to quantize the second tower's outputs. If False, effectively turns the model
            to the 'basic' model, but with different state representation
        """
        super().__init__()
        self.relu = nn.ReLU()

        # Shared state embedding (representation)
        self.state_embed = nn.Embedding(dict_size, state_embed_size, max_norm=1)
        if embed_conv_size is not None:
            self.conv_embed = nn.Conv2d(in_channels=state_embed_size, out_channels=embed_conv_size, kernel_size=3,
                                        padding=1)
        else:
            self.conv_embed = None
        self.embed_size = embed_size
        self.eps = eps
        self.commitment_cost = commitment_cost
        self.distance_loss_coef = distance_loss_coef
        self.dssm_embed_loss_coef = dssm_embed_loss_coef
        self.dssm_z_loss_coef = dssm_z_loss_coef
        self.do_quantize = do_quantize

        # phi_1
        in_channels = embed_conv_size or state_embed_size
        # batch * in_channels * height * width
        self.phi1_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        # batch * 16 * height * width
        self.phi1_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # batch * 32 * height * width -> flatten
        self.phi1_linear = nn.Linear(32 * height * width, embed_size)

        # phi_2 - same architecture as phi_1
        self.phi2_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.phi2_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.phi2_linear = nn.Linear(32 * height * width, embed_size)

        # z vectors - quantized embeddings (quantization vectors)
        self.z_vectors = nn.Parameter(torch.randn(n_z, embed_size))

        # alpha (temperature scale)
        self.scale = torch.nn.Parameter(torch.ones((1,)))

    @property
    def z_vectors_norm(self):
        """Normalised quantisation vectors"""
        return normalized(self.z_vectors, eps=0)

    def embed(self, s):
        """Generates an embedding (representation) of a state (observation) for later use in the towers"""
        x = self.state_embed(s).permute(0, 3, 1, 2)
        if self.conv_embed:
            x = self.conv_embed(x)
        return x

    def phi1(self, s):
        """The first DSSM 'tower'. Generates an embedding of a state

        :param s: state observation (or representation)
        :return: embedding
        """
        x1 = self.relu(self.phi1_conv1(s))
        x1 = self.relu(self.phi1_conv2(x1))
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.phi1_linear(x1)
        embed1 = normalized(x1, self.eps)
        return embed1

    def phi2(self, diff):
        """The second DSSM 'tower'. Generates a (quantised) embedding of a state transition.

        :param diff: state transition (difference of two state representations)
        :return: embedding
        """
        x2 = self.relu(self.phi2_conv1(diff))
        x2 = self.relu(self.phi2_conv2(x2))
        x2 = torch.flatten(x2, start_dim=1)
        x2 = self.phi2_linear(x2)
        embed2 = normalized(x2, self.eps)
        return embed2

    def forward(self, x, downscale_factor=1):
        """

        :param x: tuple of s and s' batches of some batch_size
        :param downscale_factor: should be equal to n_negatives + 1 for 'synthetic' datasets
        :return: torch tensor of shape (batch_size, batch_size) when downscale_factor = 1 or
            (synthetic batch size, downscale_factor) for downscale_factor > 1 - pairwise fitness scores
            (refer to calculate_inner_product_matrix function)
        """
        s, s_prime = x
        s_embed = self.embed(s)
        s_prime_embed = self.embed(s_prime)

        embed1 = self.phi1(s_embed)
        embed2 = self.phi2(s_prime_embed - s_embed)

        # Quantize
        if self.do_quantize:
            z_vectors_norm = self.z_vectors_norm
            z_inds = torch.argmax(torch.matmul(embed2, z_vectors_norm.T), dim=1)
            z_matrix = z_vectors_norm[z_inds]
        else:
            z_matrix = embed2

        # Calculate inner products (Gram matrix)
        inner_product_matrix = calculate_inner_product_matrix(embed1, z_matrix, downscale_factor)

        # Apply (positive) temperature scaling
        output = torch.exp(self.scale) * inner_product_matrix

        return output

    def forward_and_loss(self, x, criterion, target, downscale_factor=1):
        """Calculates both output of the model and the losses.

        :param x: tuple of s and s' batches of some batch_size
        :param criterion: loss function
        :param target: target for the loss function
        :param downscale_factor: should be equal to n_negatives + 1 for 'synthetic' datasets
        :return: dict with output and losses (total loss, quantization loss, FPS-loss) and also counts of vectors
            clipped to each of the quantization vectors (for statistics)
        """
        if self.do_quantize:
            s, s_prime = x
            s_embed = self.embed(s)
            s_prime_embed = self.embed(s_prime)

            embed1 = self.phi1(s_embed)
            embed2 = self.phi2(s_prime_embed - s_embed)

            # Quantize
            z_vectors_norm = self.z_vectors_norm
            z_inds = torch.argmax(torch.matmul(embed2, z_vectors_norm.T), dim=1)
            z_matrix = z_vectors_norm[z_inds]

            encoder_latent_loss = ((z_matrix.detach() - embed2) ** 2).mean() * self.embed_size
            quant_latent_loss = ((z_matrix - embed2.detach()) ** 2).mean() * self.embed_size
            total_loss = self.distance_loss_coef * (quant_latent_loss + self.commitment_cost * encoder_latent_loss)

            # Straight Through Estimator
            # Gradients flow to phi2 parameters
            z_matrix_from_embed = embed2 + (z_matrix - embed2).detach()

            # Calculate inner products (Gram matrix)
            inner_product_matrix_from_embed = calculate_inner_product_matrix(embed1, z_matrix_from_embed,
                                                                             downscale_factor)

            # Apply (positive) temperature scaling
            output_from_embed = torch.exp(self.scale) * inner_product_matrix_from_embed

            dssm_loss_from_embed = criterion(output_from_embed, target)
            total_loss += dssm_loss_from_embed * self.dssm_embed_loss_coef

            if self.dssm_z_loss_coef is not None:
                # Here gradients will flow to z_vectors
                inner_product_matrix_from_z = calculate_inner_product_matrix(embed1, z_matrix, downscale_factor)
                output_from_z = torch.exp(self.scale) * inner_product_matrix_from_z
                dssm_loss_from_z = criterion(output_from_z, target)
                total_loss += dssm_loss_from_z * self.dssm_z_loss_coef

            z_inds_count = torch.bincount(z_inds, minlength=len(self.z_vectors))

            results = dict(output=output_from_embed, total_loss=total_loss, encoder_latent_loss=encoder_latent_loss,
                           dssm_loss=dssm_loss_from_embed, z_inds_count=z_inds_count)

            return results
        else:
            # No quantization
            output = self.forward(x, downscale_factor)
            total_loss = criterion(output, target)
            results = dict(output=output, total_loss=total_loss, encoder_latent_loss=torch.Tensor([0]),
                           dssm_loss=total_loss, z_inds_count=torch.Tensor([1]))
            return results


# The following model is under development
def run_modules(x, modules, last_activation=False):
    if modules is None:
        return x
    for module in modules[:-1]:
        x = nn.functional.relu(module(x))
    x = modules[-1](x)
    if last_activation:
        x = nn.functional.relu(x)
    return x


class DSSMReverse(nn.Module):
    def __init__(self, dict_size=14, height=5, width=5, embed_size=64, state_embed_size=3, embed_conv_channels=None,
                 phi_conv_channels=(16, 32), fc_sizes=None, do_normalize=False, n_z=5, eps=1e-4, commitment_cost=0.25,
                 distance_loss_coef=1., dssm_embed_loss_coef=1., dssm_z_loss_coef=None, do_quantize=True):
        super().__init__()
        self.relu = nn.ReLU()
        phi_conv_channels = list(phi_conv_channels)

        # state embedding
        self.state_embed = nn.Embedding(dict_size, state_embed_size, max_norm=1)
        if embed_conv_channels is not None:
            in_channels_nums = [state_embed_size] + embed_conv_channels[:-1]
            self.embed_conv = nn.ModuleList(
                [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1) for
                 in_channels, out_channels in zip(in_channels_nums, embed_conv_channels)]
            )
        else:
            self.embed_conv = None

        self.embed_size = embed_size
        self.eps = eps
        self.do_normalize = do_normalize
        self.commitment_cost = commitment_cost
        self.distance_loss_coef = distance_loss_coef
        self.dssm_embed_loss_coef = dssm_embed_loss_coef
        self.dssm_z_loss_coef = dssm_z_loss_coef
        self.do_quantize = do_quantize

        # phi_1
        in_channels = embed_conv_channels[-1] if embed_conv_channels else state_embed_size
        # batch * in_channels * height * width
        in_channels_nums = [in_channels] + phi_conv_channels[:-1]
        self.phi1_conv = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1) for
             in_channels, out_channels in zip(in_channels_nums, phi_conv_channels)]
        )
        # batch * channels * height * width -> flatten
        self.phi1_linear = nn.Linear(phi_conv_channels[-1] * height * width, embed_size)

        # phi_2 - same architecture as phi_1
        self.phi2_conv = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1) for
             in_channels, out_channels in zip(in_channels_nums, phi_conv_channels)]
        )
        self.phi2_linear = nn.Linear(phi_conv_channels[-1] * height * width, embed_size)

        # z vectors - quantized embeddings
        self.z_vectors = nn.Parameter(torch.randn(n_z, embed_size))

        # Fully connected
        if fc_sizes:
            in_features_num = [embed_size] + fc_sizes[:-1]
            self.fc_layers = nn.ModuleList(
                [nn.Linear(in_features=in_features, out_features=out_features) for in_features, out_features in
                 zip(in_features_num, fc_sizes)]
            )
        else:
            self.fc_layers = None

        # phi_3 - same as phi_1 (or phi_2) + fully connected
        self.phi3_conv = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1) for
             in_channels, out_channels in zip(in_channels_nums, phi_conv_channels)]
        )
        self.phi3_linear = nn.Linear(phi_conv_channels[-1] * height * width, embed_size)
        if fc_sizes:
            self.phi3_fc = nn.ModuleList(
                [nn.Linear(in_features=in_features, out_features=out_features) for in_features, out_features in
                 zip(in_features_num, fc_sizes)]
            )
        else:
            self.phi3_fc = None

        # alpha (temperature scale)
        self.scale = torch.nn.Parameter(torch.ones((1,)))

    def embed(self, s):
        x = self.state_embed(s).permute(0, 3, 1, 2)
        x = run_modules(x, self.embed_conv, last_activation=True)
        return x

    def phi1(self, x):
        x = run_modules(x, self.phi1_conv, last_activation=True)
        x = torch.flatten(x, start_dim=1)
        x = self.phi1_linear(x)
        if self.do_normalize:
            x = normalized(x)
        return x

    def phi2(self, x):
        x = run_modules(x, self.phi2_conv, last_activation=True)
        x = torch.flatten(x, start_dim=1)
        x = self.phi2_linear(x)
        if self.do_normalize:
            x = normalized(x)
        return x

    def get_z_vectors(self):
        if self.do_normalize:
            return normalized(self.z_vectors)
        else:
            return self.z_vectors

    def quantize(self, x):
        z_vectors = self.get_z_vectors()
        batch_z = torch.unsqueeze(z_vectors, dim=0)
        batch_x = torch.unsqueeze(x, dim=0)
        # Batched pairwise distance
        dist = torch.squeeze(torch.cdist(batch_x, batch_z, p=2.), dim=0)
        z_inds = torch.argmax(dist, dim=1)
        z_matrix = z_vectors[z_inds]
        return {'z_matrix': z_matrix, 'z_inds': z_inds}

    def fc(self, x):
        x = run_modules(x, self.fc_layers, last_activation=False)
        x = normalized(x, self.eps)
        return x

    def phi3(self, x):
        x = run_modules(x, self.phi3_conv, last_activation=True)
        x = torch.flatten(x, start_dim=1)
        x = self.phi3_linear(x)
        x = run_modules(x, self.phi3_fc, last_activation=False)
        x = normalized(x, self.eps)
        return x

    def forward(self, x, downscale_factor):
        s, s_prime = x
        s_embed = self.embed(s)
        s_prime_embed = self.embed(s_prime)

        s_intermediate = self.phi1(s_embed)
        diff_intermediate = self.phi2(s_prime_embed - s_embed)
        s_prime_out = self.phi3(s_prime_embed)

        # quantize
        if self.do_quantize:
            diff_quant = self.quantize(diff_intermediate)
        else:
            diff_quant = diff_intermediate

        s_out = self.fc(s_intermediate + diff_quant)

        # calculate inner products (Gram matrix)
        inner_product_matrix = calculate_inner_product_matrix(s_out, s_prime_out, downscale_factor)

        # apply (positive) temperature scaling
        output = torch.exp(self.scale) * inner_product_matrix

        return output

    def forward_and_loss(self, x, criterion, target, downscale_factor=1):
        if self.do_quantize:
            s, s_prime = x
            s_embed = self.embed(s)
            s_prime_embed = self.embed(s_prime)

            s_intermediate = self.phi1(s_embed)
            diff_intermediate = self.phi2(s_prime_embed - s_embed)
            s_prime_out = self.phi3(s_prime_embed)

            # quantize
            results = self.quantize(diff_intermediate)
            diff_quant = results['z_matrix']
            z_inds = results['z_inds']

            encoder_latent_loss = ((diff_quant.detach() - diff_intermediate) ** 2).mean() * self.embed_size
            quant_latent_loss = ((diff_quant - diff_intermediate.detach()) ** 2).mean() * self.embed_size
            total_loss = self.distance_loss_coef * (quant_latent_loss + self.commitment_cost * encoder_latent_loss)

            # Straight Through Estimator
            # Gradients flow to phi2 parameters
            diff_quant_from_embed = diff_intermediate + (diff_quant - diff_intermediate).detach()
            s_out_from_embed = self.fc(s_intermediate + diff_quant_from_embed)

            # calculate inner products (Gram matrix)
            inner_product_matrix_from_embed = calculate_inner_product_matrix(s_out_from_embed, s_prime_out,
                                                                             downscale_factor)

            # apply (positive) temperature scaling
            output_from_embed = torch.exp(self.scale) * inner_product_matrix_from_embed

            dssm_loss_from_embed = criterion(output_from_embed, target)
            total_loss += dssm_loss_from_embed * self.dssm_embed_loss_coef

            if self.dssm_z_loss_coef is not None:
                # Here gradients will flow to z_vectors
                s_out_from_z = self.fc(s_intermediate + diff_quant)
                inner_product_matrix_from_z = calculate_inner_product_matrix(s_out_from_z, s_prime_out,
                                                                             downscale_factor)
                output_from_z = torch.exp(self.scale) * inner_product_matrix_from_z
                dssm_loss_from_z = criterion(output_from_z, target)
                total_loss += dssm_loss_from_z * self.dssm_z_loss_coef

            results = dict(output=output_from_embed, total_loss=total_loss, encoder_latent_loss=encoder_latent_loss,
                           dssm_loss=dssm_loss_from_embed)

            z_inds_count = torch.bincount(z_inds, minlength=len(self.z_vectors))
            results['z_inds_count'] = z_inds_count

            return results
        else:
            output = self.forward(x, downscale_factor)
            total_loss = criterion(output, target)
            results = dict(output=output, total_loss=total_loss, dssm_loss=total_loss)
            return results
