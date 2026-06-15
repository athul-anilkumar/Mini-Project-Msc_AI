import pandas as pd
import matplotlib.pyplot as plt
import glob

print("Generating Traffic Graph...")

# 1. Find the CSV file dynamically
csv_files = glob.glob("outputs/baseline_fixed_timer*.csv")

if not csv_files:
    print("Error: Could not find the baseline CSV file.")
else:
    file_path = csv_files[0]
    df = pd.read_csv(file_path)
    
    # 2. Plot the System Mean Waiting Time
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['system_mean_waiting_time'], label='Fixed-Timer Baseline', color='red', linewidth=2)
    
    # 3. Format the graph
    plt.title('Traffic Grid Wait Times over 1 Hour', fontsize=14)
    plt.xlabel('Simulation Step (Seconds)', fontsize=12)
    plt.ylabel('System Mean Waiting Time (Seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 4. Save and display
    plt.savefig('outputs/baseline_graph.png')
    print("Graph saved to outputs/baseline_graph.png")
    plt.show()