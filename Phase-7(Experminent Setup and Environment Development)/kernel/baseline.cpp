#include <iostream>
#include <vector>

/*
 * Baseline Benchmark: The Unpacking Tax
 * This simulates what happens when an edge chip is forced to run standard INT8 MACs
 * instead of a bit-serial optimized kernel.
 */
int main() {
    std::cout << "Starting Baseline Simulation (Standard NPU)..." << std::endl;
    int accumulator = 0;
    
    // Simulate unpacking compressed ternary to 8-bit, then executing heavy 8x8 multipliers
    for(int i = 0; i < 16384; i++) {
        // Arbitrary math to force the ALU to perform work and generate clock cycles
        int8_t weight = (i % 3) - 1; // {-1, 0, 1}
        int8_t activation = (i % 127);
        
        accumulator += weight * activation;
    }
    
    std::cout << "Baseline MAC Output: " << accumulator << std::endl;
    return 0;
}
