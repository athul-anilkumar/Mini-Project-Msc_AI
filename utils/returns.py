import numpy as np


def compute_returns(rewards, gamma=0.99):

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = reward + gamma * G

        returns.insert(0, G)

    return returns