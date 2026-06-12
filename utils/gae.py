import numpy as np


def compute_gae(
    rewards,
    values,
    dones,
    gamma=0.99,
    lam=0.95
):

    advantages = []

    gae = 0

    values = values + [0]

    for step in reversed(range(len(rewards))):

        delta = (
            rewards[step]
            + gamma * values[step + 1] * (1 - dones[step])
            - values[step]
        )

        gae = (
            delta
            + gamma * lam * (1 - dones[step]) * gae
        )

        advantages.insert(0, gae)

    return advantages