import numpy as np

from mpe2 import simple_spread_v3

from algorithms.mappo import MAPPO


NUM_EPISODES = 20
ROLLOUT_LENGTH = 50


env = simple_spread_v3.parallel_env()

obs, infos = env.reset()

agent = MAPPO(
    obs_dim=len(obs["agent_0"]),
    action_dim=env.action_space(
        "agent_0"
    ).n,
    num_agents=len(env.agents)
)

episode_rewards = []

for episode in range(NUM_EPISODES):

    obs, infos = env.reset()

    total_reward = 0

    for step in range(ROLLOUT_LENGTH):

        if len(obs) == 0:
            break

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

        if len(rewards) > 0:

            team_reward = np.mean(
                list(rewards.values())
            )

            total_reward += team_reward

            dones = {}

            for agent_id in rewards.keys():

                terminated = terminations.get(
                    agent_id,
                    False
                )

                truncated = truncations.get(
                    agent_id,
                    False
                )

                dones[agent_id] = (
                    terminated
                    or truncated
                )

            agent.store_transition(
                obs,
                global_obs,
                actions,
                log_probs,
                rewards,
                dones,
                value
            )

        if len(next_obs) == 0:
            break

        obs = next_obs

    print(
        f"\nEpisode {episode + 1}"
    )

    print(
        f"Reward: {total_reward:.2f}"
    )

    print(
        f"Buffer Size Before Update: {agent.buffer.size()}"
    )

    result = agent.update()

    print(
        f"Actor Loss: {result['actor_loss']:.4f}"
    )

    print(
        f"Critic Loss: {result['critic_loss']:.4f}"
    )

    print(
        f"Total Loss: {result['total_loss']:.4f}"
    )

    print(
        f"Buffer Size After Update: {agent.buffer.size()}"
    )

    episode_rewards.append(
        total_reward
    )

print("\nTraining Complete")

print("\nEpisode Rewards:")

for i, reward in enumerate(
    episode_rewards,
    start=1
):

    print(
        f"Episode {i}: {reward:.2f}"
    )