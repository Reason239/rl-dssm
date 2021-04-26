import torch
from torch import nn


class DSSM(nn.Module):
    def __init__(self, in_channels=7, height=5, width=5, embed_size=64):
        super().__init__()
        self.relu = nn.ReLU()

        # phi_1
        # batch * height * width * in_channels
        self.phi1_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        # batch * height * width * 16
        self.phi1_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # batch * height * width * 32 -> flatten
        self.phi1_linear = nn.Linear(height * width * 32, embed_size)
        # TODO try just norm (may be eps=1)
        self.phi1_norm = nn.LayerNorm(normalized_shape=embed_size)

        # phi_2 - same architecture as phi_1
        self.phi2_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.phi2_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.phi2_linear = nn.Linear(height * width * 32, embed_size)
        self.phi2_norm = nn.LayerNorm(normalized_shape=embed_size)

        # alpha (temperature scale)
        self.scale = torch.nn.Parameter(torch.ones((1,)))

    def phi1(self, s):
        x1 = self.relu(self.phi1_conv1(s))
        x1 = self.relu(self.phi1_conv2(x1))
        x1 = torch.flatten(x1, start_dim=1)
        embed1 = self.phi1_norm(self.phi1_linear(x1))
        return embed1

    def phi2(self, diff):
        x2 = self.relu(self.phi2_conv1(diff))
        x2 = self.relu(self.phi2_conv2(x2))
        x2 = torch.flatten(x2, start_dim=1)
        embed2 = self.phi2_norm(self.phi2_linear(x2))
        return embed2

    def forward(self, x):
        s, s_prime = x
        # s, s_prime = embed(s), embed(s_prime)

        embed1 = self.phi1(s)

        # TODO what if just s_prime. Will easily see the button configuration
        embed2 = self.phi2(s_prime - s)

        # calculate inner products (Gramm matrix)
        gramm = torch.matmul(embed1, torch.transpose(embed2, 0, 1))
        output = self.scale * gramm

        return output

    def forward_and_loss(self, x, criterion, target):
        output = self.forward(x)
        loss = criterion(output, target)
        return output, loss


def normalized(matrix, eps):
    return matrix / (torch.sqrt(torch.sum(matrix ** 2, dim=1, keepdim=True)) + eps)


class DSSMEmbed(nn.Module):
    def __init__(self, dict_size=14, height=5, width=5, embed_size=64, state_embed_size=3, embed_conv_channels=None,
                 n_z=5, eps=1e-4, commitment_cost=0.25):
        super().__init__()
        self.relu = nn.ReLU()

        # state embedding
        self.state_embed = nn.Embedding(dict_size, state_embed_size, max_norm=1)
        if embed_conv_channels is not None:
            self.conv_embed = nn.Conv2d(in_channels=state_embed_size, out_channels=embed_conv_channels, kernel_size=3,
                                        padding=1)
        else:
            self.conv_embed = None
        self.eps = eps
        self.commitment_cost = commitment_cost

        # phi_1
        in_channels = embed_conv_channels or state_embed_size
        # batch * height * width * in_channels
        self.phi1_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        # batch * height * width * 16
        self.phi1_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # batch * height * width * 32 -> flatten
        self.phi1_linear = nn.Linear(height * width * 32, embed_size)

        # phi_2 - same architecture as phi_1
        self.phi2_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.phi2_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.phi2_linear = nn.Linear(height * width * 32, embed_size)

        # z vectors
        self.z_vectors = nn.Parameter(torch.randn(n_z, embed_size))

        # alpha (temperature scale)
        self.scale = torch.nn.Parameter(torch.ones((1,)))

    def embed(self, s):
        x = self.state_embed(s).permute(0, 3, 1, 2)
        if self.conv_embed:
            x = self.conv_embed(x)
        return x

    def phi1(self, s):
        x1 = self.relu(self.phi1_conv1(s))
        x1 = self.relu(self.phi1_conv2(x1))
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.phi1_linear(x1)
        embed1 = normalized(x1, self.eps)
        return embed1

    def phi2(self, diff):
        x2 = self.relu(self.phi2_conv1(diff))
        x2 = self.relu(self.phi2_conv2(x2))
        x2 = torch.flatten(x2, start_dim=1)
        x2 = self.phi2_linear(x2)
        embed2 = normalized(x2, self.eps)
        return embed2

    def forward(self, x):
        s, s_prime = x
        s, s_prime = self.embed(s), self.embed(s_prime)

        embed1 = self.phi1(s)
        embed2 = self.phi2(s_prime - s)

        z_vectors_norm = normalized(self.z_vectors, eps=0)
        z_inds = torch.argmax(torch.matmul(embed2, z_vectors_norm.T), dim=1)
        z_matrix = z_vectors_norm[z_inds]

        # calculate inner products (Gramm matrix)
        gramm = torch.matmul(embed1, z_matrix.T)
        output = torch.exp(self.scale) * gramm

        return output

    def forward_and_loss(self, x, criterion, target):
        s, s_prime = x
        s, s_prime = self.embed(s), self.embed(s_prime)

        embed1 = self.phi1(s)
        embed2 = self.phi2(s_prime - s)

        # quantize
        z_vectors_norm = normalized(self.z_vectors, 0)
        z_inds = torch.argmax(torch.matmul(embed2, z_vectors_norm.T), dim=1)
        z_matrix = z_vectors_norm[z_inds]

        encoder_latent_loss = ((z_matrix.detach() - embed2) ** 2).mean()
        quant_latent_loss = ((z_matrix - (embed2).detach()) ** 2).mean()
        loss = quant_latent_loss + self.commitment_cost * encoder_latent_loss

        # Straight Through Estimator
        z_matrix = embed2 + (z_matrix - embed2).detach()

        # calculate inner products (Gramm matrix)
        gramm = torch.matmul(embed1, z_matrix.T)
        output = torch.exp(self.scale) * gramm
        loss += criterion(output, target)

        return output, loss
