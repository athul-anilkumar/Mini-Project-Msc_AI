import argparse
from sumo_rl import parallel_env
import warnings

warnings.filterwarnings("ignore")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    # Launch SUMO environment on a unique port
    env = parallel_env(
        net_file=args.net,
        route_file='my_routes.rou.xml',
        use_gui=True,
        num_seconds=3600,
        sumo_port=args.port # This prevents the port collision!
    )
    
    obs, _ = env.reset()
    while env.agents:
        # Here you would load your specific trained model for this net
        actions = {agent: 0 for agent in env.agents} # Placeholder
        _, _, _, _, _ = env.step(actions)
    
    env.close()

if __name__ == "__main__":
    run()