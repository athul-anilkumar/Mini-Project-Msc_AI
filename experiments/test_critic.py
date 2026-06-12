import torch

from mpe2 import simple_spread_v3
from networks.critic import Critic

env = simple_spread_v3.parallel_env()

obs, infos = env.reset()

sample_obs = obs["agent_0"]

obs_dim = sample_obs.shape[0]

critic = Critic(obs_dim)

input_tensor = torch.FloatTensor(sample_obs).unsqueeze(0)

value = critic(input_tensor)

print("State Value Shape:", value.shape)

print(value)