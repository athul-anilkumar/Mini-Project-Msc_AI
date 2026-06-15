from buffers.multi_agent_buffer import (
    MultiAgentBuffer
)

buffer = MultiAgentBuffer()

buffer.store(

    obs={
        "agent_0": [1, 2],
        "agent_1": [3, 4]
    },

    global_obs=[1, 2, 3, 4],

    actions={
        "agent_0": 0,
        "agent_1": 1
    },

    log_probs={
        "agent_0": -0.5,
        "agent_1": -0.8
    },

    rewards={
        "agent_0": 2,
        "agent_1": 1
    },

    dones={
        "agent_0": False,
        "agent_1": False
    },

    value=5.2

)

print(buffer.size())

print(buffer.actions)

print(buffer.rewards)