import torch
import torch.optim as optim
import numpy as np

from torch.distributions import Categorical

from networks.actor import Actor
from networks.centralized_critic import CentralizedCritic

from buffers.multi_agent_buffer import MultiAgentBuffer

from utils.returns import compute_returns
from utils.gae import compute_gae

from algorithms.ppo_loss import compute_actor_loss
from algorithms.value_loss import compute_value_loss


class MAPPO:

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_agents
    ):

        self.num_agents = num_agents

        self.actor = Actor(
            obs_dim,
            action_dim
        )

        self.critic = CentralizedCritic(
            obs_dim * num_agents
        )

        self.buffer = MultiAgentBuffer()

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=3e-4
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=3e-4
        )

    def get_actions(self, observations):

        actions = {}
        log_probs = {}

        for agent, obs in observations.items():

            obs_tensor = torch.FloatTensor(
                obs
            ).unsqueeze(0)

            logits = self.actor(
                obs_tensor
            )

            dist = Categorical(
                logits=logits
            )

            action = dist.sample()

            actions[agent] = (
                action.item()
            )

            log_probs[agent] = (
                dist.log_prob(action)
                .item()
            )

        return actions, log_probs

    def get_value(self, observations):

        global_obs = np.concatenate(
            [
                observations[agent]
                for agent in observations
            ]
        )

        state_tensor = torch.FloatTensor(
            global_obs
        ).unsqueeze(0)

        value = self.critic(
            state_tensor
        )

        return (
            value.item(),
            global_obs
        )

    def store_transition(
        self,
        obs,
        global_obs,
        actions,
        log_probs,
        rewards,
        dones,
        value
    ):

        self.buffer.store(
            obs,
            global_obs,
            actions,
            log_probs,
            rewards,
            dones,
            value
        )

    def extract_rewards(self):

        rewards = []

        for i, step_rewards in enumerate(
            self.buffer.rewards
        ):

            if len(step_rewards) == 0:

                print(
                    f"WARNING: Empty reward dict at step {i}"
                )

                continue

            avg_reward = np.mean(
                list(step_rewards.values())
            )

            if np.isnan(avg_reward):

                print(
                    f"WARNING: NaN reward at step {i}"
                )

                continue

            rewards.append(avg_reward)

        return rewards

    def extract_values(self):

        return self.buffer.values

    def compute_buffer_returns(self):

        rewards = self.extract_rewards()

        if len(rewards) == 0:

            return [0.0]

        returns = compute_returns(
            rewards
        )

        return returns

    def compute_buffer_advantages(self):

        rewards = self.extract_rewards()

        values = self.extract_values()

        dones = []

        for step_done in self.buffer.dones:

            done = any(
                step_done.values()
            )

            dones.append(
                int(done)
            )

        min_len = min(
            len(rewards),
            len(values),
            len(dones)
        )

        rewards = rewards[:min_len]

        values = values[:min_len]

        dones = dones[:min_len]

        print(
            f"Rewards={len(rewards)} "
            f"Values={len(values)} "
            f"Dones={len(dones)}"
        )

        advantages = compute_gae(
            rewards,
            values,
            dones
        )

        return advantages

    def prepare_training_batch(self):

        obs_batch = []

        action_batch = []

        old_log_prob_batch = []

        for step in range(
            len(self.buffer.obs)
        ):

            obs_dict = (
                self.buffer.obs[step]
            )

            action_dict = (
                self.buffer.actions[step]
            )

            log_prob_dict = (
                self.buffer.log_probs[step]
            )

            for agent in obs_dict:

                obs_batch.append(
                    obs_dict[agent]
                )

                action_batch.append(
                    action_dict[agent]
                )

                old_log_prob_batch.append(
                    log_prob_dict[agent]
                )

        return (
            torch.FloatTensor(
                np.array(obs_batch)
            ),
            torch.LongTensor(
                action_batch
            ),
            torch.FloatTensor(
                old_log_prob_batch
            )
        )

    def update(self):

        print("Inside update()")

        actor_loss = (
            self.compute_actor_update_loss()
        )

        critic_loss = (
            self.compute_critic_update_loss()
        )

        if torch.isnan(actor_loss):

            print(
                "ERROR: Actor loss is NaN"
            )

            return {
                "actor_loss": float("nan"),
                "critic_loss": float("nan"),
                "total_loss": float("nan")
            }

        if torch.isnan(critic_loss):

            print(
                "ERROR: Critic loss is NaN"
            )

            return {
                "actor_loss": float("nan"),
                "critic_loss": float("nan"),
                "total_loss": float("nan")
            }

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        print("Clearing buffer...")

        self.buffer.clear()

        print("Buffer cleared.")

        return {

            "actor_loss":
            actor_loss.item(),

            "critic_loss":
            critic_loss.item(),

            "total_loss":
            actor_loss.item()
            + critic_loss.item()
        }
    def compute_new_log_probs(
         self,
         obs_batch,
         action_batch
         ):

     logits = self.actor(
          obs_batch
         )

     dist = Categorical(
        logits=logits
         )

     new_log_probs = (
         dist.log_prob(
            action_batch
          )
        )

     return new_log_probs
    def compute_actor_update_loss(self):

     obs_batch, action_batch, old_log_probs = (
          self.prepare_training_batch()
        )

     advantages = (
        self.compute_buffer_advantages()
         )

     advantages = torch.FloatTensor(
        advantages
         )

     advantages = advantages.repeat_interleave(
         self.num_agents
         )

     new_log_probs = (
         self.compute_new_log_probs(
            obs_batch,
            action_batch
            )
        )

     actor_loss = compute_actor_loss(
        new_log_probs,
        old_log_probs,
        advantages
        )

     return actor_loss
    
    def train_actor(self):

        actor_loss = (
            self.compute_actor_update_loss()
        )

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        return actor_loss.item()
    def compute_critic_update_loss(self):

        returns = (
            self.compute_buffer_returns()
        )

        returns = torch.FloatTensor(
            returns
        )

        global_obs_batch = torch.FloatTensor(
            np.array(self.buffer.global_obs)
        )

        predicted_values = (
            self.critic(
                global_obs_batch
            )
            .squeeze()
        )

        critic_loss = (
            compute_value_loss(
                predicted_values,
                returns
            )
        )

        return critic_loss
    def train_critic(self):

        critic_loss = (
            self.compute_critic_update_loss()
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        return critic_loss.item()
    
    def extract_rewards(self):

        rewards = []

        for step_rewards in self.buffer.rewards:

            avg_reward = np.mean(
                list(step_rewards.values())
            )

            rewards.append(avg_reward)

        return rewards
    
    def compute_buffer_advantages(self):

        rewards = self.extract_rewards()

        values = self.extract_values()

        dones = []

        for step_done in self.buffer.dones:

            done = any(
                step_done.values()
            )

            dones.append(
                int(done)
            )

        advantages = compute_gae(
            rewards,
            values,
            dones
        )

        return advantages