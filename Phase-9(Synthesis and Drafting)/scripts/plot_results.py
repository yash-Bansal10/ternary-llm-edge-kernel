import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please place the CSV file in this folder.")
        return

    # Read the data
    df = pd.read_csv(csv_path)

    # Set the visual style for academic papers
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # ==========================================
    # 1. Bar Chart: Baseline vs Optimized Cycles
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Restructure data for easier plotting
    df_melted = pd.melt(df, id_vars=['Clock_Speed'], 
                        value_vars=['Baseline_Cycles', 'Optimized_Cycles'],
                        var_name='Model', value_name='Total_Cycles')
    
    # Rename labels for aesthetics
    df_melted['Model'] = df_melted['Model'].replace({
        'Baseline_Cycles': 'Baseline: Unpacked INT8 Vector Loop',
        'Optimized_Cycles': 'Optimized: Fused Bit-Serial Popcount Kernel (Ours)'
    })

    ax = sns.barplot(x='Clock_Speed', y='Total_Cycles', hue='Model', data=df_melted, palette=['#e74c3c', '#2ecc71'])
    
    plt.title('Inference Latency Comparison: Baseline vs Fused Kernel', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Edge AI CPU Clock Frequency', fontsize=12, fontweight='bold')
    plt.ylabel('CPU Clock Cycles (in Billions)', fontsize=12, fontweight='bold')
    plt.legend(title='Architecture')
    
    # Format y-axis to show billions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}B".format(x/1e9)))
    
    plt.tight_layout()
    plt.savefig('../image/latency_comparison_barchart.png', dpi=300)
    print("✅ Successfully generated: ../image/latency_comparison_barchart.png")

    # ==========================================
    # 2. Line Graph: Speedup Percentage
    # ==========================================
    plt.figure(figsize=(8, 5))
    
    # Convert string percentages to floats
    df['Speedup_Float'] = df['Latency_Reduction'].str.rstrip('%').astype('float')
    
    sns.lineplot(x='Clock_Speed', y='Speedup_Float', data=df, marker='o', color='#3498db', linewidth=2.5, markersize=10)
    
    plt.title('Latency Reduction Across Hardware Tiers', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Edge AI CPU Clock Frequency', fontsize=12, fontweight='bold')
    plt.ylabel('Relative Latency Reduction (%)', fontsize=12, fontweight='bold')
    plt.ylim(0, 30)
    
    # Add data labels to the points
    for x, y, label in zip(df['Clock_Speed'], df['Speedup_Float'], df['Latency_Reduction']):
        plt.text(x, y + 1, label, ha='center', va='bottom', fontweight='bold', color='#2980b9')

    plt.tight_layout()
    plt.savefig('../image/latency_reduction_linegraph.png', dpi=300)
    print("✅ Successfully generated: ../image/latency_reduction_linegraph.png")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "..", "data", "sweep_results_pivoted.csv")
    plot_results(csv_file)
