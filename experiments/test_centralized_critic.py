import torch
import numpy as np

from mpe2 import simple_spread_v3

from networks.centralized_critic import (
    CentralizedCritic
)

env = simple_spread_v3.parallel_env()

obs, infos = env.reset()

global_obs = np.concatenate(
    [obs[agent] for agent in env.agents]
)

global_obs_dim = len(global_obs)

critic = CentralizedCritic(
    global_obs_dim
)

input_tensor = torch.FloatTensor(
    global_obs
).unsqueeze(0)

value = critic(input_tensor)

print("Global Observation Shape:",
      global_obs.shape)

print("Value Shape:",
      value.shape)

print(value)