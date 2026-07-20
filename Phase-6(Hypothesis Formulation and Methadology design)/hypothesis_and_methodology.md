# Phase 6: Hypothesis Formulation and Methodology Design

**Research Topic:** Extreme Quantization and Model Compression for Edge AI
**Core Focus:** Bypassing the INT8 Unpacking Tax and ALU Underutilization using Software-level Bit-Serial Popcount Kernels

---

## 1. The Core Hypothesis

Despite the massive theoretical memory and energy savings proposed by **1.58-bit (Ternary) Large Language Models**, executing these networks on standard commodity edge devices (which are heavily optimized for INT8 arithmetic) results in severe computational bottlenecks. 

**We hypothesize that:**
1.  **The ALU Underutilization & Unpacking Tax:** Standard INT8 Multiply-Accumulate (MAC) NPU circuits operate at ~15-20% efficiency when multiplying by ternary weights `{-1, 0, 1}`. Furthermore, unpacking these tightly compressed 2-bit weights into an 8-bit format that the NPU can read costs 4-8 CPU cycles per weight, effectively destroying the speed advantages of extreme quantization.
2.  **The Bit-Serial Bypass:** By abandoning the NPU's INT8 multiplier entirely and processing the INT8 activations sequentially (bit-serially), we can emulate matrix multiplication using primitive **Bitwise AND** logic. We can then emulate the addition tree using native **Hardware Popcount**. 
3.  **The Expected Outcome:** This software-only kernel will yield **100% mathematically equivalent results** to standard matrix multiplication while physically bypassing the NPU MAC bottlenecks, enabling extreme quantization on off-the-shelf edge hardware without the need for millions of dollars in custom ASIC/silicon manufacturing.

---

## 2. Methodology Design (How We Prove It)

To scientifically validate our hypothesis, we will employ a four-step experimental methodology:

### Step A: Baseline Construction (The Control Group)
We will develop a standard C++ baseline that mimics how standard neural network frameworks (like TensorFlow Lite or PyTorch Mobile) handle ternary weights on edge devices:
*   Store ternary weights in a dual-bitplane compressed format.
*   Write an explicit loop to unpack these 2-bit weights into `int8_t` variables.
*   Execute standard `INT8 * INT8` Multiply-Accumulate (MAC) operations.
*   *Objective:* Isolate and benchmark the exact latency cost of the "Unpacking Tax."

### Step B: Optimized Kernel Development (The Experimental Group)
We will develop the proposed bit-serial execution kernel:
*   Pre-pack the 8-bit activations into 8 separate bit-planes (simulating ideal bit-serial hardware buffering).
*   Process the dot-product using bitwise logic (`pos_plane & activation_bitplane`) to replace multiplication.
*   Use native `__builtin_popcount` to replace the massive accumulator tree.
*   Shift and sum the results mathematically according to their binary significance (Two's Complement).

### Step C: Algorithmic Equivalence Validation
Before any performance claims can be made, the math must be perfect. We will run both the Control and Experimental kernels over large randomized matrices (e.g., $N=16,384$) of dummy weights and activations.
*   *Success Metric:* The final output integer of the baseline loop and the optimized bit-serial loop must match with 0% deviation.

### Step D: Cross-Architecture Performance Benchmarking
Once mathematical equivalency is proven, we will wrap the execution loops in high-resolution timers (`std::chrono`) to capture execution latency (microseconds per dot-product). 
*   We will initially benchmark on a standard high-performance CPU architecture to prove code viability (Phase 7 & 8).
*   We will evaluate how these metrics map to constrained edge architectures (e.g., ARM Cortex-M or RISC-V) where the INT8 multiplier pipeline is a massive bottleneck compared to desktop x86 CPUs.

---

## 3. Scope and Constraints

To ensure this research is distinct from existing publications (like `xTern` or ASIC proposals), our methodology strictly enforces the following constraints:
*   **Software-Only:** We will not propose any new physical hardware designs (no FPGAs, no custom silicon, no Compute-in-Memory proposals).
*   **No Custom ISA Extensions:** The solution must rely entirely on universally available instructions (standard bitwise operators and native Popcount). We will not assume the existence of custom RISC-V ternary extensions.
*   **Edge-Focused:** The focus is on latency reduction and memory bandwidth relief specifically for low-power edge microcontrollers and NPUs, not server-grade GPUs.
