import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_scaling():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    plt.figure(figsize=(9, 6))
    
    # Simulate Model Sizes from 1 Billion to 10 Billion Parameters
    model_sizes = np.array([1, 2, 4, 7, 10]) # in Billions
    
    # Theoretical Baseline Scaling (Steep linear growth due to memory bottleneck)
    # Based on standard 8-bit fetching
    baseline_latency = model_sizes * 1.5  
    
    # Theoretical Optimized Scaling (Flatter growth due to 2-bit packing)
    # Using our ~22% experimental reduction as a conservative baseline, 
    # but theoretically it scales even better as memory bottlenecks worsen at larger sizes.
    optimized_latency = model_sizes * 1.15
    
    plt.plot(model_sizes, baseline_latency, marker='o', color='#e74c3c', linewidth=3, markersize=8, label='Standard INT8 Execution Path')
    plt.plot(model_sizes, optimized_latency, marker='s', color='#2ecc71', linewidth=3, markersize=8, label='Bit-Serial Execution Path (Ours)')
    
    # Fill the gap to show the massive savings
    plt.fill_between(model_sizes, optimized_latency, baseline_latency, color='#2ecc71', alpha=0.1, label='Latency Savings')
    
    plt.title('Projected Latency Scaling vs. Model Size', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Model Parameters (Billions)', fontsize=13, fontweight='bold')
    plt.ylabel('Projected Execution Latency', fontsize=13, fontweight='bold')
    
    # Remove y-ticks since this is a theoretical projection
    plt.yticks([]) 
    
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    
    plt.savefig('../image/theoretical_scaling_linegraph.png', dpi=300)
    print("✅ Successfully generated: ../image/theoretical_scaling_linegraph.png")

if __name__ == "__main__":
    plot_scaling()
