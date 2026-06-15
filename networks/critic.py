import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        value = self.network(x)
        return value