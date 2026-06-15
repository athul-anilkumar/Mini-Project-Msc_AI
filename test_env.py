from sumo_rl import parallel_env

print("Initializing Environment Data Extraction...")

def custom_reward_function(traffic_signal):
    """
    A custom reward prioritizing high throughput while preventing side-street starvation.
    """
    # 1. Get the current queue length across all incoming lanes
    total_queue = traffic_signal.get_total_queued()
    
    # 2. Get the total waiting time of all cars currently at the intersection (in seconds)
    # This function returns a list of wait times per lane, so we sum it up.
    total_wait = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    
    # 3. Apply the weights
    # We divide the wait time by 100 because wait times can quickly reach into the thousands,
    # which would overpower the queue penalty and destabilize the neural network.
    reward = -(total_queue + (total_wait / 100.0))
    
    return reward

# 1. Load the environment (GUI turned off for speed)
env = parallel_env(
    net_file='my_grid.net.xml',
    route_file='my_routes.rou.xml',
    use_gui=False, 
    num_seconds=3600,
    reward_fn=custom_reward_function
)

# 2. Reset to get the starting state
observations, infos = env.reset()

# 3. Get the ID of the first traffic light (e.g., 't_0')
first_agent_id = env.agents[0]

# 4. Extract and print the exact PyTorch dimensions your colleague needs
state_vector = observations[first_agent_id]
action_space = env.action_space(first_agent_id)

print("\n" + "="*50)
print("🚦 DATA CONTRACT FOR MARL ENGINEER 🚦")
print("="*50)
print(f"Agent ID: {first_agent_id}")
print(f"State Vector (Raw Data): {state_vector}")
print(f"State Dimension (PyTorch Input Nodes): {len(state_vector)}")
print(f"Action Space (PyTorch Output Nodes): {action_space.n}")
print("="*50 + "\n")

# Clean up
env.close()