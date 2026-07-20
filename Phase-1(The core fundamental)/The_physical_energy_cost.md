Neural Network Weight Quantization: FP32 → INT8

 ### Technical Analysis of Memory Bandwidth, Memory Footprint, and ALU Utilization

 ────────────────────────────────────────────────────────────────────────────────

 ### 1. Memory Footprint

 Bit-width Reduction:
```
 ┌────────┬────────────────┬─────────────────────┐
 │ Format │ Bits per Value │ Storage Requirement │
 ├────────┼────────────────┼─────────────────────┤
 │ FP32   │ 32 bits        │ 4 bytes per weight  │
 ├────────┼────────────────┼─────────────────────┤
 │ INT8   │ 8 bits         │ 1 byte per weight   │
 └────────┴────────────────┴─────────────────────┘
```
 Quantitative Impact:

 - Linear memory reduction: 4× — A model requiring N FP32 parameters consumes 4N bytes; quantized to INT8, it consumes N bytes (Krishnamoorthi et al., 2018; Nagel et al., 2021, arXiv:2106.08295).
 - Effective memory hierarchy utilization:
     - L1/L2 cache capacity: 4× more weights fit per cache line (32-byte cache line holds 8 INT8 values vs 2 FP32 values)
     - TLB coverage: 4× more addressable parameter space per TLB entry
     - DRAM allocation: Direct 4× reduction in off-chip memory allocation

 Edge device implications:

 On memory-constrained edge processors (e.g., ARM Cortex-M series with 256KB–2MB SRAM), FP32 models often exceed on-chip SRAM capacity, forcing off-chip DRAM access. INT8 quantization can convert DRAM-bound models to SRAM-resident models, eliminating off-chip memory traffic entirely.

 Example: MobileNetV2 (3.4M parameters)
 - FP32: 13.6 MB (exceeds typical MCU SRAM)
 - INT8: 3.4 MB (fits in external Flash/SRAM, partial on-chip residency possible)

 ────────────────────────────────────────────────────────────────────────────────

 ### 2. Memory Bandwidth

 Data Transfer Reduction:

 Memory bandwidth (GB/s) is the rate at which data moves between memory hierarchy levels. Quantization directly reduces bandwidth demand:

 $$\text{Bandwidth}{\text{INT8}} = \frac{\text{Bandwidth}{\text{FP32}}}{4}$$

 Roofline Model Analysis (from Jouppi et al., ISCA 2021):

 The roofline model characterizes whether a workload is compute-bound or memory-bound based on operational intensity (OI):

 $$OI = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$$

 For a matrix multiplication $Y = W \times X$ where $W \in \mathbb{R}^{M \times K}$, $X \in \mathbb{R}^{K \times N}$:

 - FP32: Each weight fetch transfers 4 bytes
 - INT8: Each weight fetch transfers 1 byte

 Operational intensity impact:
```
 ┌────────────────────────┬───────────┬───────────┬─────────────┐
 │ Operation              │ FP32 OI   │ INT8 OI   │ Change      │
 ├────────────────────────┼───────────┼───────────┼─────────────┤
 │ Weight-stationary GEMM │ 0.25 OP/B │ 1.0 OP/B  │ 4× increase │
 ├────────────────────────┼───────────┼───────────┼─────────────┤
 │ Activation fetch       │ Unchanged │ Unchanged │ —           │
 └────────────────────────┴───────────┴───────────┴─────────────┘
```
 The increased operational intensity moves the operating point rightward on the roofline diagram, potentially transitioning from memory-bound to compute-bound regime.

 Energy per memory access (Jouppi et al., ISCA 2021, Table 2):
```
 ┌─────────────┬──────────────────────────────────┐
 │ Memory Type │ Energy (pJ/64-bit access) at 7nm │
 ├─────────────┼──────────────────────────────────┤
 │ SRAM (8KB)  │ 7.5 pJ                           │
 ├─────────────┼──────────────────────────────────┤
 │ SRAM (32KB) │ 8.5 pJ                           │
 ├─────────────┼──────────────────────────────────┤
 │ SRAM (1MB)  │ 14 pJ                            │
 ├─────────────┼──────────────────────────────────┤
 │ HBM2        │ 250–450 pJ                       │
 ├─────────────┼──────────────────────────────────┤
 │ DDR3/4      │ 1300 pJ                          │
 └─────────────┴──────────────────────────────────┘
```
 Bandwidth-energy relationship:

 $$E_{\text{memory}} = B \times E_{\text{per-byte}} \times N_{\text{accesses}}$$

 Where B is bytes per access. INT8 quantization reduces B by 4× for weight accesses, directly reducing energy per inference by up to 4× for weight-dominated memory traffic.

 Cache efficiency:

 INT8 improves cache line utilization:

 - FP32: 64-byte cache line holds 16 weights
 - INT8: 64-byte cache line holds 64 weights

 Spatial locality improvement: 4× more weights per cache line means fewer cache misses per layer execution.

 ────────────────────────────────────────────────────────────────────────────────

 ### 3. ALU Utilization

 3.1 Arithmetic Operation Complexity

 Integer vs. Floating-Point Unit Design:
```
 ┌────────────────┬─────────────────────────────────────────────────────┬────────────────────────────┐
 │ Operation      │ FP32 Hardware                                       │ INT8 Hardware              │
 ├────────────────┼─────────────────────────────────────────────────────┼────────────────────────────┤
 │ Addition       │ Mantissa alignment + add + normalize + round        │ Direct add                 │
 ├────────────────┼─────────────────────────────────────────────────────┼────────────────────────────┤
 │ Multiplication │ 24-bit mantissa multiply + exponent add + normalize │ 8-bit integer multiply     │
 ├────────────────┼─────────────────────────────────────────────────────┼────────────────────────────┤
 │ Accumulation   │ 24-bit mantissa with guard bits                     │ 32-bit accumulator (INT32) │
 └────────────────┴─────────────────────────────────────────────────────┴────────────────────────────┘

 Energy per operation (Jouppi et al., ISCA 2021, Table 2 at 7nm):

 ┌───────────┬───────────┬───────────┬───────┐
 │ Operation │ FP32 (pJ) │ INT8 (pJ) │ Ratio │
 ├───────────┼───────────┼───────────┼───────┤
 │ ADD       │ 1.31      │ 0.07      │ 18.7× │
 ├───────────┼───────────┼───────────┼───────┤
 │ MUL       │ 3.70      │ 0.21      │ 17.6× │
 └───────────┴───────────┴───────────┴───────┘
```
 3.2 Throughput Improvement: SIMD Vectorization

 ARM Cortex-M4/M7 DSP Extensions (ARM, 2016):

 The Cortex-M4/M7 implement SIMD instructions operating on multiple data elements within a single 32-bit register:

 ```
   ┌─────────────────────────────────┐
   │      32-bit Register            │
   ├──────────┬──────────┬──────────┬──────────┤
   │  INT8[3] │  INT8[2] │  INT8[1] │  INT8[0] │
   └──────────┴──────────┴──────────┴──────────┘
            ↓ SIMD QADD8 instruction ↓
   ┌──────────┬──────────┬──────────┬──────────┐
   │   add    │   add    │   add    │   add    │
   └──────────┴──────────┴──────────┴──────────┘
      (4 parallel 8-bit additions in 1 cycle)
 ```

 INT8 SIMD throughput:
 - QADD8/QSUB8: 4 operations/cycle
 - FP32 VADD: 1 operation/cycle (Cortex-M7 FPU)

 Throughput ratio: 4× for INT8 SIMD vs FP32 scalar

 3.3 Systolic Array Utilization (Google TPU Architecture)

 TPUv1 Systolic Array (Jouppi et al., 2017):

 - 256×256 systolic array = 65,536 MAC units
 - INT8 weights × INT8 activations → INT32 accumulation

 Matrix multiplication compute density:

 For $Y = W \times X$ where $W \in \mathbb{R}^{M \times K}$, $X \in \mathbb{R}^{K \times N}$:

 $$\text{Total MACs} = M \times K \times N$$

 FP32 vs INT8 in systolic arrays:

 From Nagel et al. (2021, arXiv:2106.08295):

 │ "The computational cost for matrix multiplication decreases quadratically by a factor of 16."

 Derivation:

 For a 256×256 systolic array:
```
 ┌────────┬───────────────────┬──────────────┬──────────────────┐
 │ Format │ MAC Unit Area     │ MACs per mm² │ Peak TOPS        │
 ├────────┼───────────────────┼──────────────┼──────────────────┤
 │ FP32   │ ~1.0 (normalized) │ 1×           │ 92 TFLOPS (FP32) │
 ├────────┼───────────────────┼──────────────┼──────────────────┤
 │ INT8   │ ~0.25 (4× denser) │ 4×           │ 92 TOPS (INT8)   │
 └────────┴───────────────────┴──────────────┴──────────────────┘
```
 However, the quadratic factor of 16 arises from:

 $$\text{Speedup}_{\text{effective}} = \frac{\text{Compute Density} \times \text{Throughput}}{\text{FP32 baseline}}$$

 The 16× factor accounts for both:
 1. 4× from bit-width reduction (more MACs per unit area)
 2. 4× from reduced latency (simpler integer arithmetic enables higher clock rates or deeper pipelining)

 TPUv4i Improvements (Jouppi et al., ISCA 2021):
```
 ┌────────────────────┬───────┬────────┬─────────────┐
 │ Metric             │ TPUv3 │ TPUv4i │ Improvement │
 ├────────────────────┼───────┼────────┼─────────────┤
 │ MXU count per core │ 2     │ 4      │ 2×          │
 ├────────────────────┼───────┼────────┼─────────────┤
 │ Peak TFLOPS (bf16) │ 123   │ 138    │ 1.1×        │
 ├────────────────────┼───────┼────────┼─────────────┤
 │ Peak TOPS (INT8)   │ —     │ 138    │ —           │
 ├────────────────────┼───────┼────────┼─────────────┤
 │ SRAM capacity      │ 32 MB │ 144 MB │ 4.5×        │
 └────────────────────┴───────┴────────┴─────────────┘
```
 The TPUv4i architecture includes 4 MXUs per core, each optimized for both bf16 and INT8 arithmetic, enabling flexible precision selection while maximizing ALU utilization.

 3.4 Accumulation Precision

 INT8 × INT8 → INT32 Accumulation:

 Standard quantized inference uses INT8 weights and activations with INT32 accumulation:

 ```
   INT8 weight × INT8 activation
            ↓
       INT16 product (max magnitude: 127 × 127 = 16,129)
            ↓
       INT32 accumulator (prevents overflow across ~130K accumulations)
            ↓
       INT8 output (requantized with scale/zero-point)
 ```

 Accumulator bit-width requirement:

 $$\text{Accumulator bits} = 8 + 8 + \lceil \log_2(K) \rceil$$

 Where K is the reduction dimension. For K=1024: $8 + 8 + 10 = 26$ bits → INT32 (32 bits) sufficient.

 FP32 comparison:

 FP32 multiplication produces a 24-bit mantissa product, requiring full 32-bit floating-point accumulation with guard bits. INT8 accumulation uses integer addition, which is:
 - Simpler hardware: No denormal handling, no NaN propagation, no rounding modes
 - Lower latency: Integer add is combinatorial; FP add requires alignment, addition, normalization

 ────────────────────────────────────────────────────────────────────────────────

 ### 4. Quantitative Summary

 Memory footprint:
 $$\text{Storage}{\text{INT8}} = \frac{\text{Storage}{\text{FP32}}}{4}$$

 Memory bandwidth:
 $$\text{Bandwidth}{\text{required, INT8}} = \frac{\text{Bandwidth}{\text{required, FP32}}}{4}$$

 Energy per inference (memory-dominated):
 $$E_{\text{INT8}} \approx \frac{E_{\text{FP32}}}{4} \text{ (weight access energy)}$$

 Energy per inference (compute-dominated):
 $$E_{\text{INT8, compute}} \approx \frac{E_{\text{FP32, compute}}}{18.7} \text{ (ADD)} \times \frac{1}{17.6} \text{ (MUL)}$$

 ALU throughput:
 - ARM Cortex-M SIMD: 4× (4 parallel INT8 operations vs 1 FP32 operation)
 - Systolic arrays: Up to 16× effective throughput (quadratic improvement)

 Operational intensity:
 $$OI_{\text{INT8}} = 4 \times OI_{\text{FP32}}$$

 ────────────────────────────────────────────────────────────────────────────────

 ### Sources

 1. Nagel, M., et al. "A White Paper on Neural Network Quantization." Qualcomm AI Research, arXiv:2106.08295, 2021. https://arxiv.org/abs/2106.08295
 2. Wu, H., et al. "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation." arXiv:2004.09602, 2020. https://arxiv.org/abs/2004.09602
 3. Jouppi, N. P., et al. "Ten Lessons From Three Generations Shaped Google's TPUv4i." ISCA 2021. https://arxiv.org/pdf/2106.08295
 4. ARM Limited. "The DSP capabilities of ARM Cortex-M4 and Cortex-M7 Processors." White Paper, 2016. https://developer.arm.com/documentation/
 5. Krishnamoorthi, R. "Quantizing deep convolutional networks for efficient inference: A whitepaper." arXiv:1806.08342, 2018.