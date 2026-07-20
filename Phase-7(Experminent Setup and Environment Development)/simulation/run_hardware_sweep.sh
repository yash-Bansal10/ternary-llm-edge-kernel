#!/bin/bash
echo " Starting Parameter Sweep Simulation..."

for clk in "500MHz" "1GHz" "2GHz"
do
  echo "=========================================="
  echo "  Running Baseline Simulation at ${clk}..."
  # Assumes gem5 is installed one directory above Phase-7, adjust if needed
  ../../gem5/build/ARM/gem5.opt -d "m5out_baseline_${clk}" edge_config.py ../kernel/baseline_arm.bin "${clk}"
  
  echo "  Running Optimized Simulation at ${clk}..."
  ../../gem5/build/ARM/gem5.opt -d "m5out_optimized_${clk}" edge_config.py ../kernel/optimized_arm.bin "${clk}"
done

echo "=========================================="
echo " Parameter Sweep Complete! Extracting Metrics..."
python3 ../scripts/extract_metrics.py
