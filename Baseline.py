import os
import warnings
# Import the native base environment instead of the PettingZoo wrapper
from sumo_rl import SumoEnvironment

warnings.filterwarnings("ignore", module="pettingzoo.utils.conversions")

print("Starting Baseline (Fixed-Timer) Data Collection...")

# 1. Create an outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# 2. Initialize the native environment
env = SumoEnvironment(
    net_file='my_grid.net.xml',
    route_file='my_routes.rou.xml',
    use_gui=False, 
    num_seconds=3600,
    out_csv_name='outputs/baseline_fixed_timer'
)

env.reset()

step_count = 0

# 3. Run the simulation using the internal timer
# This avoids any unpacking errors completely!
while env.sim_step < env.sim_max_time:
    # An empty dictionary tells the native environment to let SUMO run the lights
    env.step({})
    step_count += 1
env.save_csv("outputs/baseline_fixed_timer", 1)

env.close()
print(f"Baseline collection complete! Simulated {step_count} steps.")
print("Check the 'outputs' folder for your CSV file.")