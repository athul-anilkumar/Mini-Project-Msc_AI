import argparse
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, required=True, help="The map file")
    parser.add_argument("--port", type=int, required=True, help="TraCI port")
    parser.add_argument("--mode", type=str, default="ai", choices=["ai", "fixed"], help="AI or Fixed Timer")
    args = parser.parse_args()

    # 1. The Mapping Dictionaries
    model_map = {
        "grid2x2.net.xml": "model_2x2.pth",
        "grid3x3.net.xml": "model_3x3.pth",
        "grid4x4.net.xml": "model_4x4.pth",
        "grid5x5.net.xml": "model_5x5.pth"
    }
    
    route_map = {
        "grid2x2.net.xml": "routes_2x2.rou.xml",
        "grid3x3.net.xml": "routes_3x3.rou.xml",
        "grid4x4.net.xml": "routes_4x4.rou.xml",
        "grid5x5.net.xml": "routes_5x5.rou.xml"
    }

    route_file = route_map.get(args.net)
    if not route_file:
        print(f"Error: No route mapping found for {args.net}")
        return

    history_rewards = []
    
    # ==========================================
    # BRANCH A: FIXED TIMER BASELINE LOGIC
    # ==========================================
    if args.mode == "fixed":
        import traci
        import sumolib
        print(f"[{args.net}] Starting FIXED TIMER Baseline...")
        
        sumo_binary = sumolib.checkBinary('sumo-gui')
        sumo_cmd = [sumo_binary, "-n", args.net, "-r", route_file]
        traci.start(sumo_cmd)
        
        step = 0
        while step < 3600 and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            # Calculate total system waiting time to mirror the AI's reward logic
            wait_time = sum([traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList()])
            # We track negative wait time so "Higher" on the graph is still better
            history_rewards.append(-wait_time) 
            step += 1
            
        traci.close()
        print(f"[{args.net}] Baseline Simulation complete.")
        plot_color = 'crimson'
        plot_label = 'Standard Fixed Timer'

    # ==========================================
    # BRANCH B: MAPPO AI LOGIC
    # ==========================================
    else:
        from sumo_rl import parallel_env
        from algorithms.mappo import MAPPO
        
        brain_file = model_map.get(args.net)
        if not brain_file:
            print(f"Error: No trained model mapping found for {args.net}")
            return
            
        print(f"[{args.net}] Starting MAPPO AI environment...")
        env = parallel_env(net_file=args.net, route_file=route_file, use_gui=True, num_seconds=3600)
        obs, _ = env.reset()
        
        first_agent = env.agents[0]
        agent = MAPPO(obs_dim=len(obs[first_agent]), action_dim=env.action_space(first_agent).n, num_agents=len(env.agents))
        
        try:
            agent.actor.load_state_dict(torch.load(brain_file))
            agent.actor.eval()
        except FileNotFoundError:
            print(f"CRITICAL ERROR: Could not find {brain_file}. Did you train it?")
            env.close()
            return

        print(f"[{args.net}] AI has taken control of the traffic lights!")
        while env.agents:
            actions, _ = agent.get_actions(obs)
            next_obs, rewards, _, _, _ = env.step(actions)
            
            # Track the total system reward
            history_rewards.append(sum(rewards.values()))
            
            if not next_obs:
                break
            obs = next_obs
        
        env.close()
        print(f"[{args.net}] AI Simulation complete.")
        plot_color = 'teal'
        plot_label = 'MAPPO Agent'

    # ==========================================
    # GENERATE ANALYTICS GRAPH
    # ==========================================
    print(f"[{args.net}] Generating analytics...")
    plt.figure(figsize=(8, 5))
    
    window = 50 
    if len(history_rewards) > window:
        smoothed_data = np.convolve(history_rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed_data, label=plot_label, color=plot_color, linewidth=2)
    else:
        plt.plot(history_rewards, label=plot_label, color=plot_color)
        
    plt.title(f'Traffic Synchronization Performance\nLayout: {args.net} | Mode: {plot_label}', fontsize=14, fontweight='bold')
    plt.xlabel('Simulation Steps', fontsize=12)
    plt.ylabel('System Reward (Higher = Less Wait Time)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    
    graph_filename = f"Graph_{args.mode}_{args.net.replace('.net.xml', '')}.png"
    plt.savefig(graph_filename, dpi=300)
    print(f"[{args.net}] Graph saved as {graph_filename}")
    plt.show()

if __name__ == "__main__":
    run()