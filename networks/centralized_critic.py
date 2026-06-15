import torch
import torch.nn as nn


class CentralizedCritic(nn.Module):

    def __init__(self, global_obs_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(global_obs_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 1)

        )

    def forward(self, x):

        return self.network(x)