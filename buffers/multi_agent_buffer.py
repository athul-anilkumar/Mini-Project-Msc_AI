class MultiAgentBuffer:

    def __init__(self):

        self.clear()

    def clear(self):

        self.obs = []

        self.global_obs = []

        self.actions = []

        self.log_probs = []

        self.rewards = []

        self.dones = []

        self.values = []

    def store(
        self,
        obs,
        global_obs,
        actions,
        log_probs,
        rewards,
        dones,
        value
    ):

        self.obs.append(obs)

        self.global_obs.append(global_obs)

        self.actions.append(actions)

        self.log_probs.append(log_probs)

        self.rewards.append(rewards)

        self.dones.append(dones)

        self.values.append(value)

    def size(self):

        return len(self.obs)