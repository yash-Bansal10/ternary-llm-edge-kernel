import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np

def plot_results(csv_path="benchmark_raw_arm64.csv", output_prefix="v5"):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)

    # Calculate medians per layer
    medians = df.groupby("layer").median(numeric_only=True).reset_index()

    # 1. Stacked Bar Chart: Pack Time vs Kernel Time
    plt.figure(figsize=(10, 6))
    sns.barplot(x="layer", y="proposed_kernel_us", data=medians, color="#1f77b4", label="Kernel Time")
    plt.bar(medians["layer"], medians["activation_pack_us"], 
            bottom=medians["proposed_kernel_us"], color="#ff7f0e", label="Pack Time")
    plt.title(f"Bit-Serial Popcount Latency Breakdown ({output_prefix.upper()})")
    plt.ylabel("Latency (Microseconds)")
    plt.xlabel("Layer Shape")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    stacked_out = f"{output_prefix}_latency_breakdown.png"
    plt.savefig(stacked_out, dpi=300)
    print(f"Saved: {stacked_out}")
    plt.close()

    # 2. Boxplot to show variance across the 60 trials
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="layer", y="proposed_total_us", hue="layer", data=df, legend=False)
    plt.title(f"Total Latency Variance Across 60 Trials ({output_prefix.upper()})")
    plt.ylabel("Total Latency (Microseconds)")
    plt.xlabel("Layer Shape")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    box_out = f"{output_prefix}_latency_variance.png"
    plt.savefig(box_out, dpi=300)
    print(f"Saved: {box_out}")
    plt.close()

    # 3. Baseline Comparison Chart (If baselines exist)
    if "baseline_a_us" in df.columns and "baseline_b_us" in df.columns:
        plt.figure(figsize=(12, 7))
        
        layers = medians["layer"]
        x = np.arange(len(layers))
        width = 0.25
        
        plt.bar(x - width, medians["baseline_a_us"], width, label='Baseline A (Dense MAC Limit)', color='lightgray')
        plt.bar(x, medians["baseline_b_us"], width, label='Baseline B (Naive Scalar Unpacking)', color='salmon')
        plt.bar(x + width, medians["proposed_total_us"], width, label='Proposed (Bit-Serial + Pack)', color='teal')
        
        plt.title(f"Speedup Comparison vs Baselines ({output_prefix.upper()})")
        plt.ylabel("Latency (Microseconds) - Lower is Better")
        plt.xlabel("Layer Shape")
        plt.xticks(x, layers)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        speedup_out = f"{output_prefix}_speedup_comparison.png"
        plt.savefig(speedup_out, dpi=300)
        print(f"Saved: {speedup_out}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot V5 Benchmark Results")
    parser.add_argument("--csv", default="benchmark_raw_arm64.csv", help="Path to the CSV file")
    parser.add_argument("--prefix", default="v5", help="Prefix for output images")
    args = parser.parse_args()
    
    plot_results(args.csv, args.prefix)
