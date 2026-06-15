import numpy as np
from sumo_rl import parallel_env
from algorithms.mappo import MAPPO
import warnings
import torch

warnings.filterwarnings("ignore", module="pettingzoo.utils.conversions")
# Force PyTorch to use Tensor Cores for matrix multiplications
torch.set_float32_matmul_precision('high')
# 1. Bring in YOUR custom reward function
def custom_reward_function(traffic_signal):
    total_queue = traffic_signal.get_total_queued()
    total_wait = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    # Penalize queue and wait time (scaled down)
    return -(total_queue + (total_wait / 100.0))

print("Initializing SUMO Environment with MAPPO Agent...")

# 2. Swap the dummy env for YOUR SUMO env
env = parallel_env(
    net_file='my_grid.net.xml',
    route_file='my_routes.rou.xml',
    use_gui=False, # Turn to False later for fast overnight training
    num_seconds=3600,
    reward_fn=custom_reward_function,
    out_csv_name='outputs/mappo_training_run'
)

# Training Hyperparameters
NUM_EPISODES = 20
ROLLOUT_LENGTH = 100 # Increased slightly for traffic flow

obs, infos = env.reset()

# SUMO names its agents 't_0', 't_1', etc. (Not 'agent_0')
first_agent = env.agents[0] 

# 3. Initialize your colleague's MAPPO agent dynamically
agent = MAPPO(
    obs_dim=len(obs[first_agent]),
    action_dim=env.action_space(first_agent).n,
    num_agents=len(env.agents)
)

episode_rewards = []

for episode in range(NUM_EPISODES):
    obs, infos = env.reset()
    total_reward = 0

    # 4. The Execution Loop (Your colleague's exact logic)
    for step in range(ROLLOUT_LENGTH):
        if not env.agents: # Stop if all traffic lights disappear
            break

        # AI decides the traffic lights
        actions, log_probs = agent.get_actions(obs)
        value, global_obs = agent.get_value(obs)

        # Environment steps forward
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        if len(rewards) > 0:
            team_reward = np.mean(list(rewards.values()))
            total_reward += team_reward

            dones = {a: (terminations.get(a, False) or truncations.get(a, False)) for a in rewards.keys()}

            # Save memories to buffer
            agent.store_transition(obs, global_obs, actions, log_probs, rewards, dones, value)

        if not next_obs:
            break
            
        obs = next_obs

    print(f"\nEpisode {episode + 1} | Reward: {total_reward:.2f}")

    # Train the neural network
    result = agent.update()
    print(f"Actor Loss: {result['actor_loss']:.4f} | Critic Loss: {result['critic_loss']:.4f}")

    episode_rewards.append(total_reward)

env.close()
print("\nTraining Complete! Check the outputs folder for CSV data.")