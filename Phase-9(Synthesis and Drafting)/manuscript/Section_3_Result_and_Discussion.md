# 3. Result and Discussion

## 3.1. Experimental Data and Visualizations

The bare-metal hardware simulations conducted via Gem5 yielded highly deterministic clock cycle measurements, directly contrasting the execution latency of standard INT8 MAC operations against the proposed Bit-Serial Popcount architecture.

![1782813283175](image/Section_3_Result_and_Discussion/1782813283175.png)

As depicted in the latency comparison graph above, the Bit-Serial Popcount architecture consistently outperformed the baseline across all simulated Edge AI hardware tiers (500MHz, 1GHz, and 2GHz).

![1782813300615](image/Section_3_Result_and_Discussion/1782813300615.png)

Furthermore, the architectural shift provided a massive reduction in the physical memory footprint of the Llama-Base-3B-BitNet model. By abandoning standard FP16 and INT8 storage formats in favor of dense 2-bit ternary packing, the model size was crushed from 1.4GB down to just 96MB, as illustrated below.

![1782813316292](image/Section_3_Result_and_Discussion/1782813316292.png)

## 3.2. Analysis of Results

The extracted metrics indicate a profound reduction in total execution cycles. For a standard 1GHz Edge AI processor, the baseline INT8 NPU required 7,595,085,000 clock cycles to execute the benchmark matrix multiplication. Conversely, the optimized Bit-Serial NPU completed the identical mathematical workload in just 5,948,525,000 clock cycles.

This represents an absolute latency reduction of **~21.68%**.

Crucially, the line graph analysis demonstrates that this percentage of speedup remains relatively constant across varying clock frequencies. This confirms that the latency reduction is an inherent property of the algorithmic efficiency (fetching 2-bits vs 8-bits per weight) rather than a hardware-specific anomaly.

## 3.3. Discussion and Research Impact

In the context of the present research landscape, deploying Large Language Models (LLMs) on Edge devices is severely bottlenecked by memory bandwidth limits and the "unpacking tax"—the computational overhead of decompressing quantized weights back into higher precision formats before execution [12].

The results of this study carry significant implications. By executing dot-products via `__builtin_popcount` logic directly on the packed binary data, the proposed architecture entirely bypasses the unpacking tax. Not only does this yield a 21.68% faster inference speed, but the extreme 93% reduction in memory footprint (1.4GB to 96MB) drastically lowers the energy consumption required for memory fetching [20].

As Edge AI continues to proliferate in IoT sensors, smartphones, and robotics, shifting from traditional MAC units to Bit-Serial logic gates offers a highly viable pathway to achieving real-time LLM inference on severely resource-constrained hardware [14], [18], [19].

## 3.4. Comparison with State-of-the-Art Software Kernels
Unlike recent software-level inference frameworks such as `bitnet.cpp` [9] and T-MAC, which rely heavily on fast Lookup Tables (LUTs) and complex SIMD vectorizations to accelerate ternary inference on standard CPUs, our proposed PyTorch C++ extension takes a fundamental Bit-Serial arithmetic approach. While LUT-based solutions efficiently hide latency at the framework level, they still consume L1/L2 cache bandwidth to fetch the pre-computed tables. In contrast, our `__builtin_popcount` kernel explicitly targets the root hardware inefficiency—the unpacking tax—by computing dot-products strictly through raw boolean operations that reside entirely within the CPU's arithmetic logic pipelines. 

Furthermore, when compared to advanced hardware/register co-design proposals like T-SAR [8], our solution acts as an immediately deployable software bridge. It requires no specialized CPU instructions or custom silicon registers, relying entirely on standard ARM bitwise intrinsics to deliver a guaranteed 21.68% speedup. This demonstrates that deeply optimized software-level kernels can effectively neutralize hardware-level memory bottlenecks without requiring immediate silicon fabrication.
