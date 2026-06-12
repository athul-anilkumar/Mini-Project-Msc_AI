from mpe2 import simple_spread_v3

from algorithms.mappo import MAPPO

env = simple_spread_v3.parallel_env()

obs, infos = env.reset()

agent = MAPPO(
    obs_dim=len(obs["agent_0"]),
    action_dim=env.action_space(
        "agent_0"
    ).n,
    num_agents=len(env.agents)
)

for _ in range(10):

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

advantages = (
    agent.compute_buffer_advantages()
)

print(
    "Advantages:"
)

print(
    advantages
)