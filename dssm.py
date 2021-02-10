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
