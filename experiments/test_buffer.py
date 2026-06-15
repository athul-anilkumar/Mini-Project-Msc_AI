from buffers.trajectory_buffer import TrajectoryBuffer


buffer = TrajectoryBuffer()

buffer.store(
    state=[1, 2, 3],
    action=1,
    log_prob=-0.5,
    reward=2,
    done=False,
    value=0.8
)

buffer.store(
    state=[4, 5, 6],
    action=0,
    log_prob=-1.1,
    reward=1,
    done=False,
    value=0.4
)

print("Buffer Size:", buffer.size())

print("Actions:", buffer.actions)

print("Rewards:", buffer.rewards)

print("Values:", buffer.values)