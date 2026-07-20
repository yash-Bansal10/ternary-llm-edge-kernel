import m5
from m5.objects import *
import sys

# 1. Create the virtual system
system = System()
system.clk_domain = SrcClockDomain()
clock_freq = sys.argv[2] if len(sys.argv) > 2 else '1GHz'
system.clk_domain.clock = clock_freq
system.clk_domain.voltage_domain = VoltageDomain()
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('512MB')]

# 3. Use a simple, in-order CPU (like an ARM Cortex-M or A53)
system.cpu = TimingSimpleCPU()

# 4. Create the memory bus and connect the CPU
system.membus = SystemXBar()
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports

# MUST connect system port and interrupt controller for SE mode
system.cpu.createInterruptController()
system.system_port = system.membus.cpu_side_ports

# 5. Add 512MB of simple RAM
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# 6. Tell the CPU which binary program to run
binary_path = sys.argv[1]
system.workload = SEWorkload.init_compatible(binary_path)
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
