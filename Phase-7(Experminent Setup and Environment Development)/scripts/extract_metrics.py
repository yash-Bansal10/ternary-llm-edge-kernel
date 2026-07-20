import csv
import os

def parse_stats(filepath):
    metrics = {"simSeconds": "N/A", "simTicks": "N/A", "system.cpu.dcache.overallMisses": "N/A"}
    if not os.path.exists(filepath):
        return metrics
    with open(filepath, 'r') as f:
        for line in f:
            for key in metrics.keys():
                if line.startswith(key):
                    metrics[key] = line.split()[1]
    return metrics

def run():
    print("Extracting hardware metrics from Gem5 Sweep...")
    
    clocks = ["500MHz", "1GHz", "2GHz"]
    data = []
    data.append(["Clock_Speed", "Model", "Latency_Seconds", "Total_Cycles", "D_Cache_Misses"])
    
    for clk in clocks:
        base_dir = f"m5out_baseline_{clk}/stats.txt"
        opt_dir = f"m5out_optimized_{clk}/stats.txt"
        
        base_stats = parse_stats(base_dir)
        opt_stats = parse_stats(opt_dir)
        
        data.append([clk, "Baseline (Standard MAC)", base_stats["simSeconds"], base_stats["simTicks"], base_stats["system.cpu.dcache.overallMisses"]])
        data.append([clk, "Optimized (Bit-Serial)", opt_stats["simSeconds"], opt_stats["simTicks"], opt_stats["system.cpu.dcache.overallMisses"]])
        
    with open("sweep_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
        
    print("\nExtraction complete! Check your Jupyter file browser for sweep_results.csv")
    print("================================================================")
    print("Data extracted for: 500MHz, 1GHz, and 2GHz.")
    print("================================================================")

if __name__ == "__main__":
    run()
