import torch

from mpe2 import simple_spread_v3
from networks.actor import Actor


env = simple_spread_v3.parallel_env()

obs, infos = env.reset()

sample_obs = obs["agent_0"]

obs_dim = sample_obs.shape[0]

action_dim = env.action_space("agent_0").n

print("Observation Dimension:", obs_dim)
print("Action Dimension:", action_dim)

actor = Actor(obs_dim, action_dim)

input_tensor = torch.FloatTensor(sample_obs).unsqueeze(0)

output = actor(input_tensor)

print("Network Output Shape:", output.shape)

print(output)