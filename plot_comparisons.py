import pandas as pd
import matplotlib.pyplot as plt
import glob

print("Generating Final AI vs Baseline Graph...")

# 1. Find the CSV files
baseline_files = glob.glob("outputs/baseline_fixed_timer*.csv")
ai_files = glob.glob("outputs/mappo_training_run*ep20.csv")

if not baseline_files or not ai_files:
    print("Error: Could not find both CSV files. Check your outputs folder.")
else:
    # Load the data
    df_baseline = pd.read_csv(baseline_files[0])
    df_ai = pd.read_csv(ai_files[0])
    
    # --- THE FIX: ALIGN THE DATA LENGTHS ---
    # Find the minimum length between the two files
    min_len = min(len(df_baseline), len(df_ai))
    
    # Trim both dataframes to perfectly match
    df_baseline = df_baseline.iloc[:min_len].reset_index(drop=True)
    df_ai = df_ai.iloc[:min_len].reset_index(drop=True)
    # ---------------------------------------

    # Create the comparison plot
    plt.figure(figsize=(12, 7))
    
    # Plot Baseline (Red)
    plt.plot(df_baseline['step'], df_baseline['system_mean_waiting_time'], 
             label='Standard Fixed-Timer Lights', color='red', linewidth=2.5, alpha=0.8)
    
    # Plot Trained AI (Green)
    plt.plot(df_ai['step'], df_ai['system_mean_waiting_time'], 
             label='Trained MAPPO Agent (Ep 20)', color='green', linewidth=2.5)
    
    # Format the graph
    plt.title('Traffic Grid Wait Times: AI vs. Traditional', fontsize=16, fontweight='bold')
    plt.xlabel('Simulation Step (Seconds)', fontsize=12)
    plt.ylabel('Average Waiting Time per Vehicle (Seconds)', fontsize=12)
    
    # Fill the area between the curves to highlight the time saved!
    plt.fill_between(df_baseline['step'], 
                     df_ai['system_mean_waiting_time'], 
                     df_baseline['system_mean_waiting_time'], 
                     where=(df_baseline['system_mean_waiting_time'] > df_ai['system_mean_waiting_time']), 
                     interpolate=True, color='lightgreen', alpha=0.3, label='Time Saved by AI')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Save and show
    plt.savefig('outputs/Final_Comparison.png', dpi=300, bbox_inches='tight')
    print("Masterpiece saved to outputs/Final_Comparison.png")
    plt.show()