# The Beginner's Guide to Gem5 for Edge AI Simulation

**What is Gem5?**
Gem5 is not a standard software program. It is a **Hardware Simulator**. 
When you run code on your laptop, the code runs on *your* physical Intel or AMD CPU. When you run code in Gem5, Gem5 creates a "Virtual CPU" (like a tiny ARM Cortex Edge chip) in memory, and feeds your code into that virtual chip one instruction at a time. It then counts exactly how many electrical clock cycles, cache misses, and memory bandwidth that virtual chip used.

---

## The Golden Rule of Gem5
**Gem5 (in SE Mode) simulates compiled C/C++ binaries, NOT Python scripts!**
You cannot easily run PyTorch or Python inside Gem5 because booting a full operating system with Python inside a simulator takes days. 
To test our Bit-Serial kernel against the standard FP16 baseline, we must write two standalone **C++ programs**:
1. `baseline.cpp`: A standard C++ matrix multiplication (representing the unpacking tax).
2. `optimized.cpp`: Our custom Bit-Serial Popcount logic.

---

## Step 1: Download and Build Gem5

*(You will do this on the Supercomputer).*

1. **Download the source code:**
   ```bash
   git clone https://gem5.googlesource.com/public/gem5
   cd gem5
   ```
2. **Build the Simulator for ARM:**
   We want to simulate Edge devices, which almost exclusively use ARM processors (like smartphones and Raspberry Pis). We need to compile Gem5 to understand ARM code.
   ```bash
   scons build/ARM/gem5.opt -j$(nproc)
   ```
   *(Wait 15-20 minutes for this to finish).*

---

## Step 2: Cross-Compile Your C++ Code

Because Gem5 is now configured as an ARM virtual machine, you cannot feed it a standard x86 supercomputer binary. You must **Cross-Compile** your C++ code into an ARM binary.

1. **Install the ARM Cross-Compiler**
   Because you are the `root` user in your Jupyter container, you have the absolute power to install the required packages yourself! Run this command:
   ```bash
   apt update && apt install -y g++-aarch64-linux-gnu scons build-essential
   ```
   *(Note: You do not need `sudo` because you are already root. This command will install the cross-compiler, the `scons` build tool needed for Gem5, and the standard C++ libraries).*
2. **Compile the two binaries:**
   ```bash
   aarch64-linux-gnu-g++ -O3 -static baseline.cpp -o baseline_arm.bin
   aarch64-linux-gnu-g++ -O3 -static optimized.cpp -o optimized_arm.bin
   ```
   *Note the `-static` flag: This is mandatory! It packs all necessary libraries directly into the binary so Gem5 doesn't crash looking for external Linux files.*

---

## Step 3: Create the Virtual Hardware (The Config Script)

Gem5 needs a Python script to tell it *what kind* of hardware to build. We need to tell it to build a weak Edge CPU.
Create a file named `edge_config.py` in your folder and paste this code:

```python
import m5
from m5.objects import *
import sys

# 1. Create the virtual system
system = System()

# 2. Set the clock speed to 1 GHz (Typical for Edge AI devices)
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '1GHz'
system.clk_domain.voltage_domain = VoltageDomain()

# 3. Use a simple, in-order CPU (like an ARM Cortex-M or A53)
system.cpu = TimingSimpleCPU()

# 4. Create the memory bus and connect the CPU
system.membus = SystemXBar()
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports

# 5. Add 512MB of simple RAM
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# 6. Tell the CPU which binary program to run
binary_path = sys.argv[1] # We will pass this in the terminal
process = Process()
process.cmd = [binary_path]
system.cpu.workload = process
system.cpu.createThreads()

# 7. Start the simulation!
root = Root(full_system=False, system=system)
m5.instantiate()
print(f"Beginning Edge AI Simulation for: {binary_path}")
exit_event = m5.simulate()
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
```

---

## Step 4: Run the Simulation

Now we put it all together. We will run the Gem5 engine (`gem5.opt`), give it our hardware schematic (`edge_config.py`), and feed it our ARM binary.

**1. Simulate the Baseline:**
```bash
./build/ARM/gem5.opt edge_config.py baseline_arm.bin
```
*(Wait for it to finish).*

**2. Save the Baseline Data:**
Gem5 dumps all its mathematical results into a folder called `m5out/`. Every time you run Gem5, it overwrites this folder! We must rename it.
```bash
mv m5out m5out_baseline
```

**3. Simulate our Optimized Kernel:**
```bash
./build/ARM/gem5.opt edge_config.py optimized_arm.bin
```

**4. Save the Optimized Data:**
```bash
mv m5out m5out_optimized
```

---

## Step 5: Read the Results (Proving the Thesis)

Go into the `m5out_optimized` folder and open the `stats.txt` file. This file contains thousands of lines of hardware data. 
Search (using `CTRL+F` or `grep`) for these specific lines:

1. **`simSeconds`**: Exactly how many fractions of a second the Edge chip took.
2. **`simTicks`**: The exact number of electrical clock cycles used. You want to show that the `optimized` binary used drastically fewer ticks than the `baseline` binary.
3. **`system.cpu.dcache.overallMisses`** (If you added caches to your config): Proves that our bit-serial packing saved memory bandwidth.

By comparing the `stats.txt` from the baseline against the `stats.txt` from the optimized kernel, you have indisputable mathematical proof that your algorithm speeds up Edge AI hardware!
