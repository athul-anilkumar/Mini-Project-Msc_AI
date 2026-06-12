import torch

from torch.distributions import Categorical

from mpe2 import simple_spread_v3

from networks.actor import Actor


env = simple_spread_v3.parallel_env()

obs, infos = env.reset()

sample_obs = obs["agent_0"]

obs_dim = sample_obs.shape[0]

action_dim = env.action_space("agent_0").n


actor = Actor(obs_dim, action_dim)

input_tensor = torch.FloatTensor(sample_obs).unsqueeze(0)

logits = actor(input_tensor)

print("Logits:")
print(logits)

dist = Categorical(logits=logits)

action = dist.sample()

log_prob = dist.log_prob(action)

print("\nChosen Action:")
print(action.item())

print("\nLog Probability:")
print(log_prob.item())