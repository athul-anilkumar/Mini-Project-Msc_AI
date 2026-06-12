from mpe2 import simple_spread_v3


from algorithms.mappo import MAPPO


env = simple_spread_v3.parallel_env()

obs, infos = env.reset()

obs_dim = len(
    obs["agent_0"]
)

action_dim = env.action_space(
    "agent_0"
).n

num_agents = len(
    env.agents
)

agent = MAPPO(
    obs_dim,
    action_dim,
    num_agents
)

actions, log_probs = (
    agent.get_actions(obs)
)

value, global_obs = (
    agent.get_value(obs)
)

print(
    "Actions:"
)

print(actions)

print(
    "\nLog Probs:"
)

print(log_probs)

print(
    "\nState Value:"
)

print(value)

print(
    "\nGlobal Obs Shape:"
)

print(global_obs.shape)