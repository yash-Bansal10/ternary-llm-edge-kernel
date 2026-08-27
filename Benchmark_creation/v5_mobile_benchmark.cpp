#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <numeric>

// Standard POPCOUNT macro
#define POPCOUNT __builtin_popcount

using namespace std;
using namespace std::chrono;

// Configuration based on V5 Methodology
const int WARMUP_ROUNDS = 30;
const int MEASURED_TRIALS = 60;
const int INNER_REPS = 10;
const int ACTIVATION_SETS = 64;

// Layer Shapes (Rows x Cols)
struct LayerShape {
    string name;
    int rows;
    int cols;
    int words_per_row;
};

vector<LayerShape> layers = {
    {"k_proj_640x2560", 640, 2560, 2560 / 32},
    {"q_proj_2560x2560", 2560, 2560, 2560 / 32},
    {"gate_proj_6912x2560", 6912, 2560, 2560 / 32}
};

// Simulated Packed Layer
struct PackedLayer {
    vector<vector<uint32_t>> pos_masks; // rows x words_per_row
    vector<vector<uint32_t>> neg_masks;
    int rows, cols, words_per_row;
};

// Generate Random Packed Masks (Simulating the loaded .bin file)
PackedLayer generate_random_layer(const LayerShape& shape) {
    PackedLayer layer;
    layer.rows = shape.rows;
    layer.cols = shape.cols;
    layer.words_per_row = shape.words_per_row;
    layer.pos_masks.resize(shape.rows, vector<uint32_t>(shape.words_per_row));
    layer.neg_masks.resize(shape.rows, vector<uint32_t>(shape.words_per_row));
    
    mt19937 rng(42);
    uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
    
    for(int r = 0; r < shape.rows; r++) {
        for(int w = 0; w < shape.words_per_row; w++) {
            uint32_t p = dist(rng);
            uint32_t n = dist(rng);
            // Ensure pos and neg bits don't overlap (value can't be both +1 and -1)
            n &= ~p; 
            layer.pos_masks[r][w] = p;
            layer.neg_masks[r][w] = n;
        }
    }
    return layer;
}

// Generate Random INT8 Activations (64 sets)
vector<vector<int8_t>> generate_activations(int cols) {
    vector<vector<int8_t>> acts(ACTIVATION_SETS, vector<int8_t>(cols));
    mt19937 rng(123);
    uniform_int_distribution<int> dist(-128, 127);
    for(int s = 0; s < ACTIVATION_SETS; s++) {
        for(int c = 0; c < cols; c++) {
            acts[s][c] = (int8_t)dist(rng);
        }
    }
    return acts;
}

// Global Preallocated Activation Bitplanes (8 planes x words_per_row)
vector<vector<uint32_t>> act_planes(8);

// Pack INT8 to 8 Bitplanes
void pack_activations(const vector<int8_t>& acts, int words_per_row) {
    for(int b = 0; b < 8; b++) {
        fill(act_planes[b].begin(), act_planes[b].end(), 0);
    }
    
    for(int w = 0; w < words_per_row; w++) {
        for(int bit = 0; bit < 32; bit++) {
            int c = w * 32 + bit;
            int8_t val = acts[c];
            for(int b = 0; b < 8; b++) {
                if((val >> b) & 1) {
                    act_planes[b][w] |= (1U << bit);
                }
            }
        }
    }
}

// Bit-Serial Popcount Kernel (Proposed)
void run_bit_serial_kernel(const PackedLayer& layer, vector<int>& output) {
    for(int r = 0; r < layer.rows; r++) {
        int row_acc = 0;
        for(int b = 0; b < 8; b++) {
            int plane_acc = 0;
            for(int w = 0; w < layer.words_per_row; w++) {
                int pos_count = POPCOUNT(layer.pos_masks[r][w] & act_planes[b][w]);
                int neg_count = POPCOUNT(layer.neg_masks[r][w] & act_planes[b][w]);
                plane_acc += (pos_count - neg_count);
            }
            int bit_scale = (b == 7) ? -128 : (1 << b);
            row_acc += plane_acc * bit_scale;
        }
        output[r] = row_acc;
    }
}

// Benchmark Runner
void run_benchmark() {
    cout << "========================================================\n";
    cout << "  V5 Mobile Benchmark: Bit-Serial Popcount (Cortex-A55) \n";
    cout << "========================================================\n\n";

    for(const auto& shape : layers) {
        cout << "Initializing layer: " << shape.name << "..." << flush;
        PackedLayer layer = generate_random_layer(shape);
        vector<vector<int8_t>> act_sets = generate_activations(shape.cols);
        
        // Preallocate
        for(int b = 0; b < 8; b++) act_planes[b].resize(shape.words_per_row);
        vector<int> output(shape.rows);
        cout << " Done.\n";

        // Warmup
        for(int i = 0; i < WARMUP_ROUNDS; i++) {
            pack_activations(act_sets[i % ACTIVATION_SETS], shape.words_per_row);
            run_bit_serial_kernel(layer, output);
        }

        vector<double> pack_times, kernel_times, total_times;

        // Measured Trials
        for(int t = 0; t < MEASURED_TRIALS; t++) {
            double trial_pack = 0, trial_kernel = 0, trial_total = 0;
            
            for(int r = 0; r < INNER_REPS; r++) {
                int s = (t * INNER_REPS + r) % ACTIVATION_SETS;
                
                // 1. Time Pack
                auto start = high_resolution_clock::now();
                pack_activations(act_sets[s], shape.words_per_row);
                auto end_pack = high_resolution_clock::now();
                
                // 2. Time Kernel
                auto start_k = high_resolution_clock::now();
                run_bit_serial_kernel(layer, output);
                auto end_k = high_resolution_clock::now();
                
                // 3. Direct Total (Simulated by adding for now to avoid disrupting loop)
                // In actual V5 they used interleaved timings, but this gives the raw speed.
                
                trial_pack += duration_cast<nanoseconds>(end_pack - start).count() / 1000.0;
                trial_kernel += duration_cast<nanoseconds>(end_k - start_k).count() / 1000.0;
            }
            
            pack_times.push_back(trial_pack / INNER_REPS);
            kernel_times.push_back(trial_kernel / INNER_REPS);
            total_times.push_back((trial_pack + trial_kernel) / INNER_REPS);
        }

        // Calculate Medians
        sort(pack_times.begin(), pack_times.end());
        sort(kernel_times.begin(), kernel_times.end());
        sort(total_times.begin(), total_times.end());

        double median_pack = pack_times[MEASURED_TRIALS / 2];
        double median_kernel = kernel_times[MEASURED_TRIALS / 2];
        double median_total = total_times[MEASURED_TRIALS / 2];

        cout << "Results for " << shape.name << ":\n";
        cout << "  -> Median Pack Time:   " << fixed << setprecision(2) << median_pack << " us\n";
        cout << "  -> Median Kernel Time: " << median_kernel << " us\n";
        cout << "  -> Median Total Time:  " << median_total << " us\n";
        cout << "--------------------------------------------------------\n";
    }
}

int main() {
    run_benchmark();
    return 0;
}
