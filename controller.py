import argparse
import subprocess
import threading
import time

def start_sumo_instance(net_file, port):
    """
    Launches a single SUMO instance with unique port settings.
    """
    print(f"Launching {net_file} on port {port}...")
    
    # We pass the port to the command so TraCI knows which connection to use
    cmd = [
        "python", "run_instance.py", 
        "--net", net_file, 
        "--port", str(port)
    ]
    
    # Using Popen to run asynchronously so the controller isn't blocked
    process = subprocess.Popen(cmd)
    return process

def main():
    # 1. Define your layouts and unique ports
    # Port range 8813-8820 is standard for TraCI
    configs = [
        {"net": "grid2x2.net.xml", "port": 8813},
        {"net": "grid3x3.net.xml", "port": 8814},
        {"net": "grid5x5.net.xml", "port": 8815}
    ]
    
    processes = []
    
    # 2. Launch all simulations in parallel
    for cfg in configs:
        p = start_sumo_instance(cfg["net"], cfg["port"])
        processes.append(p)
        # Small delay to prevent all 3 SUMO GUIs from fighting for focus
        time.sleep(1)
        
    print("All simulations are running! Press Ctrl+C to terminate.")

    # 3. Keep the controller alive
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nTerminating all simulations...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()