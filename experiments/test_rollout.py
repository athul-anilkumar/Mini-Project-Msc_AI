from mpe2 import simple_spread_v3

from algorithms.mappo import MAPPO


env = simple_spread_v3.parallel_env()

obs, infos = env.reset()

obs_dim = len(obs["agent_0"])

action_dim = env.action_space(
    "agent_0"
).n

num_agents = len(env.agents)

agent = MAPPO(
    obs_dim,
    action_dim,
    num_agents
)

for step in range(10):

    actions, log_probs = (
        agent.get_actions(obs)
    )

    value, global_obs = (
        agent.get_value(obs)
    )

    (
        next_obs,
        rewards,
        terminations,
        truncations,
        infos
    ) = env.step(actions)

    dones = {
        a:
        terminations[a]
        or truncations[a]
        for a in env.agents
    }

    agent.store_transition(
        obs,
        global_obs,
        actions,
        log_probs,
        rewards,
        dones,
        value
    )

    obs = next_obs

print(
    "Trajectory Length:",
    agent.buffer.size()
)