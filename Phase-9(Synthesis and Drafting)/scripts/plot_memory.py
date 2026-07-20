import matplotlib.pyplot as plt
import seaborn as sns

def plot_memory():
    # Set the visual style for academic papers
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    plt.figure(figsize=(7, 6))
    
    models = ['Standard FP16 (Baseline)', 'Ternary Packed (Ours)']
    sizes_mb = [1400, 96]
    
    # Create a striking contrast with red (bad/heavy) and green (good/light)
    ax = sns.barplot(x=models, y=sizes_mb, palette=['#e74c3c', '#2ecc71'])
    
    plt.title('Memory Footprint Reduction (Language Model)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Model Size (Megabytes)', fontsize=12, fontweight='bold')
    
    # Add data labels directly on top of the bars
    for i, v in enumerate(sizes_mb):
        ax.text(i, v + 30, f"{v} MB", ha='center', fontweight='bold', color='black', fontsize=13)
        
    plt.tight_layout()
    plt.savefig('../image/memory_compression_barchart.png', dpi=300)
    print("✅ Successfully generated: ../image/memory_compression_barchart.png")

if __name__ == "__main__":
    plot_memory()
