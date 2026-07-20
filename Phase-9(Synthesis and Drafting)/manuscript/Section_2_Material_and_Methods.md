# 2. Material and Methods

## 2.1. Materials and Environment Setup

The experimental framework was constructed using an Ubuntu-based supercomputer environment to ensure high-performance compilation and bare-metal simulation. The primary model utilized for quantization was **Llama-Base-3B-BitNet** (`herry90/llama-base-3b-bitnet`) [1], a 3-billion parameter Large Language Model deployed and managed locally using **Ollama** [6], an open-source framework designed for local LLM execution. The foundational weights were extracted from the Ollama environment in `.safetensors` format (1.4GB) to facilitate direct PyTorch manipulation. The software stack was built using Python 3.10 and PyTorch [4] for tensor manipulation, alongside the GCC ARM cross-compiler (`aarch64-linux-gnu-g++`) for native binary generation.

## 2.2. Quantization and Bit-Packing Procedure

To address the memory bandwidth bottleneck inherent in standard Edge AI architectures, the FP16 network weights were clamped into a ternary precision schema {-1, 0, 1} [22]. A custom compression pipeline (`pack_weights.py`) was developed to pack these ternary states into an ultra-dense bit-serial format. By allocating exactly 2 bits per weight, the 1.4GB FP16 model was successfully compressed into a 96MB binary payload. To guarantee the reliability of the experiment, a PyTorch-based mathematical round-trip test was conducted, ensuring absolute structural integrity with zero data loss during the bitwise packing phase.

## 2.3. Architectural Bridge and C++ Integration

To execute the proposed algorithm natively, a custom C++ engine (`poc_ternary_mac.cpp`) was engineered. This module bypassed standard INT8 Multiply-Accumulate (MAC) unpacking instructions. Instead, it utilized hardware-level `__builtin_popcount` intrinsics to calculate mathematical dot-products via pure logical `XOR` and `AND` gate operations. This C++ logic was bridged directly into the Python environment using `pybind11` [5], allowing transparent memory pointer passing between the high-level software stack and the low-level execution hardware.

## 2.4. Hardware Simulation and Data Extraction

To provide deterministic, reproducible data, bare-metal hardware validation was performed using the Gem5 Hardware Simulator [3]. The simulation topology (`edge_config.py`) was configured to mimic a standard, resource-constrained Edge AI device (analogous to a Raspberry Pi 3 or IoT sensor):

- **CPU Architecture:** TimingSimpleCPU (In-order ARM execution)
- **Memory Configuration:** 512MB DDR3_1600 RAM
- **Execution Mode:** Syscall Emulation (SE) mode

The standard INT8 logic (`baseline_arm.bin`) and custom Bit-Serial logic (`optimized_arm.bin`) were cross-compiled as statically linked ARM binaries. To ensure high experimental transparency and avoid skewing the latency metrics, disk I/O operations were excluded from the simulated environment, isolating the measurement strictly to the raw mathematical execution latency. Finally, an automated parameter sweep was conducted across three distinct CPU clock frequencies (500MHz, 1GHz, and 2GHz), with total execution clock cycles recorded and extracted for comparative analysis.
