 # Extreme Quantization & Model Compression for Edge AI
**Minimum Viable Product (MVP) - Validation Instructions**

This document provides instructions for compiling, running, and validating the C++ Proof-of-Concept (`poc_ternary_mac.cpp`) which demonstrates the computational overhead (Unpacking Tax) of executing 1.58-bit (Ternary) weights on standard INT8 hardware versus a Bit-Serial Popcount approach.

---

## 1. Prerequisites and System Requirements
Before running this script, ensure the target system meets the following requirements:

*   **C++ Compiler**: A modern C++ compiler supporting standard C++14 or higher (e.g., GCC, Clang, or MSVC).
*   **Popcount Hardware Support**: The CPU must support hardware population count instructions (e.g., `POPCNT` on x86/x64, `VCNT` on ARM). Modern CPUs (built after 2010) universally support this.
*   **Operating System**: Works cross-platform on Windows, Linux, and macOS.

**Important System Considerations (Read Before Benchmarking):**
*   **Architecture Differences**: This C++ script validates the *mathematical equivalency* and algorithmic flow of bit-serial execution. However, running this on a desktop x86 CPU (like Intel/AMD) will **not** perfectly reflect the extreme performance gap that occurs on an edge NPU (like ARM Cortex-M). Desktop CPUs have massive, deeply pipelined execution units and branch predictors that heavily mask the INT8 MAC unpacking tax. The true hardware latency gap is proven mathematically by bypassing the MAC circuit, even if x86 clock speeds obscure it.
*   **Optimization Flags**: You **must** compile the script with the `-O3` (maximum optimization) flag. Without it, the compiler will not optimize the baseline loop, resulting in highly skewed and unscientific benchmarking data.

---

## 2. Input Data Generation
You do **not** need to manually provide external input files or CSVs. 
To ensure a robust test, the script automatically generates localized random dummy data:
*   **Vector Size**: `16,384` elements (representing a large matrix-vector dot product).
*   **Ternary Weights**: Randomly generated packed 2-bit values representing `{-1, 0, 1}`.
*   **Activations**: Standard random `INT8` activations (ranging from `-128` to `127`).

The script automatically pre-packs these activations into 8 bit-planes in memory to simulate the ideal hardware buffers that a bit-serial accelerator would use.

---

## 3. How to Compile and Run the Code

Open your terminal or command prompt in the directory containing `poc_ternary_mac.cpp` and use the command appropriate for your system:

### For Windows (PowerShell / MSYS2 / MinGW)
Using GCC/G++:
```powershell
g++ poc_ternary_mac.cpp -o poc_ternary_mac.exe -O3
.\poc_ternary_mac.exe
```
Using MSVC (Visual Studio Developer Command Prompt):
```cmd
cl /O2 poc_ternary_mac.cpp
poc_ternary_mac.exe
```

### For Linux / macOS
Using GCC/Clang:
```bash
g++ poc_ternary_mac.cpp -o poc_ternary_mac -O3
./poc_ternary_mac
```

---

## 4. How to Validate the Output
When the script runs, it performs 10,000 iterations of both the baseline MAC and the optimized Bit-Serial MAC to gather stable latency metrics. 

Look at the console output for the **Correctness Check** section:

```text
[Validation] Correctness Check: 
  Baseline Output:  [Numeric Value]
  Optimized Output: [Numeric Value]
  -> Results MATCH. Computation is mathematically equivalent.
```

**Validation Criteria**:
1.  **Mathematical Validity**: Both the "Baseline Output" and the "Optimized Output" must yield the exact same integer value. If they match, it proves that substituting INT8 MAC operations with bitwise `AND` + `Popcount` does not alter or degrade the neural network's arithmetic results.
2.  **Performance Check**: The output prints latency in microseconds. The `Conclusion` section explains that the optimized kernel achieves these results without unpacking the compressed weights and by physically bypassing the INT8 multipliers.

---

## 5. Edge Cases and Potential Failures

While testing, keep the following edge cases and potential failure points in mind:

*   **Failure 1: Missing Optimization Flags (`-O0`)**
    *   **Why it fails**: If compiled in debug mode (without `-O3`), the baseline unpacking loop will perform catastrophically slowly due to unrolled memory accesses. This will artificially inflate the speedup of the optimized kernel, making the research look flawed.
*   **Failure 2: Lack of Native `__builtin_popcount`**
    *   **Why it fails**: Very old compilers or exotic microcontrollers that lack a native popcount intrinsic will emulate it using software loops. This completely destroys the latency benefits of the optimized kernel, as a software popcount is much slower than a hardware MAC.
    *   **Workaround**: Ensure the target test machine supports hardware popcount. The script uses `#ifdef _MSC_VER` to route MSVC to `__popcnt` and GCC/Clang to `__builtin_popcount`.
*   **Failure 3: Sign Extension Overflow**
    *   **Why it fails**: The dot-product of 16,384 elements can exceed the limits of standard 16-bit integers. 
    *   **Resolution**: The script safely uses `int32_t` accumulators to prevent overflow during bitwise bit-plane summation. If you modify the `VECTOR_SIZE` to be extremely large (e.g., >100 million), you may overflow the 32-bit integer, and `int64_t` would be required.
