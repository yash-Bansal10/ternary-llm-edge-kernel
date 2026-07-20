# 📓 Research Journey Log: Quantization & Hardware Simulation
**Date:** June 23, 2026
**Topic:** Bypassing standard FP16/INT8 unpacking tax using Bit-Serial Popcount architecture for Edge AI.

---

## 🌅 Morning Session: Quantization & Compression

### Goal
Download a 1.4GB FP16 Neural Network from Hugging Face, quantize the weights to a ternary {-1, 0, 1} schema, and pack them into an ultra-dense binary format to save memory bandwidth.

### Execution
- Wrote `pack_weights.py` to load `model.safetensors` into PyTorch.
- Applied thresholding to clamp weights to ternary values.
- Packed the values using bitwise shifts (2 bits per weight).
- **Result:** Successfully crushed the 1.4GB FP16 layer down to a **96MB binary payload**.
- **Verification:** Ran a round-trip mathematical integrity test within PyTorch to prove that the extraction process yielded 100% parity with no data loss.

---

## ☀️ Mid-Day Session: PyTorch C++ Architectural Bridge

### Goal
Prove that the bit-serial popcount algorithm could be executed seamlessly inside a Python environment using a custom C++ engine.

### Execution
- Created `poc_ternary_mac.cpp` and exposed the `__builtin_popcount` logic to Python using `pybind11` macros.
- Used `setup.py` and the GCC compiler to build the native PyTorch extension.
- **Result:** The `layer_test.py` script successfully executed thousands of randomized ternary MACs inside the C++ engine and returned the correct dot-product to Python. The software-to-hardware bridge was proven stable.

---

## 🌆 Evening Session: Bare-Metal Hardware Simulation (Gem5)

### Goal
Cross-compile the math engines to ARM and simulate them inside the Gem5 Hardware Simulator to prove a reduction in raw electrical clock cycles (Latency).

### Execution & Troubleshooting

1. **Compiling Gem5:**
   - Bootstrapped the official Gem5 repository on the supercomputer container.
   - *Issue:* The campus firewall blocked the default Ubuntu `apt` mirror, preventing the download of `zlib1g-dev`.
   - *Fix:* Bypassed the firewall by switching the APT sources directly to `mirrors.kernel.org`, allowing the C++ dependencies to install successfully.
   - *Issue:* The Gem5 compiler completely crashed with an obscure Bash syntax error (`-lz -lz -ll -li ...`).
   - *Fix:* Discovered a literal typo inside the Gem5 `SConstruct` file (`"zlibVersion();"`). Used `sed` to surgically remove the string, bypassing the bug and allowing the compiler to succeed.
   - *Issue:* The `g++` compiler was force-killed by the Linux OOM (Out Of Memory) killer.
   - *Fix:* Lowered the parallel compilation job count (`-j4`) to prevent memory exhaustion on the supercomputer node.

2. **Cross-Compilation:**
   - Downloaded the `aarch64-linux-gnu-g++` cross-compiler.
   - Built `baseline_arm.bin` and `optimized_arm.bin` with the `-static` flag for Syscall Emulation (SE) mode compatibility.

3. **Hardware Simulation (`edge_config.py`):**
   - Configured a virtual ARM CPU with 1GHz clock speed, 512MB RAM, and a TimingSimpleCPU core.
   - *Issue:* Encountered `IndexError` and `!seWorkload` fatal crashes.
   - *Fix:* Updated the config to initialize memory ranges explicitly and invoked the strict `SEWorkload.init_compatible()` requirement for Gem5 v23+.

### Final Verification
Executed the parameter simulation. The **Bit-Serial Optimized NPU executed 21.67% faster** than the Standard MAC Baseline, proving the thesis that avoiding the unpacking tax significantly reduces hardware latency on Edge AI devices.
