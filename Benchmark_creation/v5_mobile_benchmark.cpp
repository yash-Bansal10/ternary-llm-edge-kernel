#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <fstream>

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

// Dense Layer (For Baseline A)
struct DenseLayer {
    vector<vector<int8_t>> weights;
};

// Generate Random Packed Masks
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
            n &= ~p; 
            layer.pos_masks[r][w] = p;
            layer.neg_masks[r][w] = n;
        }
    }
    return layer;
}

// Unpack PackedLayer to DenseLayer for Baseline A
DenseLayer unpack_to_dense(const PackedLayer& packed) {
    DenseLayer dense;
    dense.weights.resize(packed.rows, vector<int8_t>(packed.cols));
    for(int r = 0; r < packed.rows; r++) {
        for(int c = 0; c < packed.cols; c++) {
            int w = c / 32;
            int bit = c % 32;
            int p = (packed.pos_masks[r][w] >> bit) & 1;
            int n = (packed.neg_masks[r][w] >> bit) & 1;
            dense.weights[r][c] = p - n;
        }
    }
    return dense;
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

// Global Preallocated Activation Bitplanes
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

// Baseline A: Dense INT8 Ternary MAC
void run_baseline_a(const DenseLayer& layer, const vector<int8_t>& acts, vector<int>& output) {
    for(int r = 0; r < layer.weights.size(); r++) {
        int acc = 0;
        for(int c = 0; c < acts.size(); c++) {
            acc += layer.weights[r][c] * acts[c];
        }
        output[r] = acc;
    }
}

// Baseline B: Scalar On-the-Fly Packed Decoding
void run_baseline_b(const PackedLayer& layer, const vector<int8_t>& acts, vector<int>& output) {
    for(int r = 0; r < layer.rows; r++) {
        int acc = 0;
        for(int c = 0; c < layer.cols; c++) {
            int w = c / 32;
            int bit = c % 32;
            int p = (layer.pos_masks[r][w] >> bit) & 1;
            int n = (layer.neg_masks[r][w] >> bit) & 1;
            int weight = p - n;
            acc += weight * acts[c];
        }
        output[r] = acc;
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
    ofstream csv_file("benchmark_raw_arm64.csv");
    csv_file << "layer,trial,baseline_a_us,baseline_b_us,activation_pack_us,proposed_kernel_us,proposed_total_us\n";

    cout << "========================================================\n";
    cout << "  V5 Mobile Benchmark: Bit-Serial Popcount (Cortex-A55) \n";
    cout << "========================================================\n\n";

    for(const auto& shape : layers) {
        cout << "Initializing layer: " << shape.name << "..." << flush;
        PackedLayer layer = generate_random_layer(shape);
        DenseLayer dense_layer = unpack_to_dense(layer);
        vector<vector<int8_t>> act_sets = generate_activations(shape.cols);
        
        for(int b = 0; b < 8; b++) act_planes[b].resize(shape.words_per_row);
        vector<int> output(shape.rows);
        cout << " Done.\n";

        // Warmup
        for(int i = 0; i < WARMUP_ROUNDS; i++) {
            run_baseline_a(dense_layer, act_sets[i % ACTIVATION_SETS], output);
            run_baseline_b(layer, act_sets[i % ACTIVATION_SETS], output);
            pack_activations(act_sets[i % ACTIVATION_SETS], shape.words_per_row);
            run_bit_serial_kernel(layer, output);
        }

        vector<double> base_a_times, base_b_times, pack_times, kernel_times, total_times;

        // Measured Trials
        for(int t = 0; t < MEASURED_TRIALS; t++) {
            double trial_base_a = 0, trial_base_b = 0, trial_pack = 0, trial_kernel = 0;
            
            for(int r = 0; r < INNER_REPS; r++) {
                int s = (t * INNER_REPS + r) % ACTIVATION_SETS;
                
                // 1. Time Baseline A
                auto start_a = high_resolution_clock::now();
                run_baseline_a(dense_layer, act_sets[s], output);
                auto end_a = high_resolution_clock::now();

                // 2. Time Baseline B
                auto start_b = high_resolution_clock::now();
                run_baseline_b(layer, act_sets[s], output);
                auto end_b = high_resolution_clock::now();

                // 3. Time Pack
                auto start_pack = high_resolution_clock::now();
                pack_activations(act_sets[s], shape.words_per_row);
                auto end_pack = high_resolution_clock::now();
                
                // 4. Time Kernel
                auto start_k = high_resolution_clock::now();
                run_bit_serial_kernel(layer, output);
                auto end_k = high_resolution_clock::now();
                
                trial_base_a += duration_cast<nanoseconds>(end_a - start_a).count() / 1000.0;
                trial_base_b += duration_cast<nanoseconds>(end_b - start_b).count() / 1000.0;
                trial_pack += duration_cast<nanoseconds>(end_pack - start_pack).count() / 1000.0;
                trial_kernel += duration_cast<nanoseconds>(end_k - start_k).count() / 1000.0;
            }
            
            base_a_times.push_back(trial_base_a / INNER_REPS);
            base_b_times.push_back(trial_base_b / INNER_REPS);
            pack_times.push_back(trial_pack / INNER_REPS);
            kernel_times.push_back(trial_kernel / INNER_REPS);
            total_times.push_back((trial_pack + trial_kernel) / INNER_REPS);

            csv_file << shape.name << "," << t << "," 
                     << (trial_base_a / INNER_REPS) << ","
                     << (trial_base_b / INNER_REPS) << ","
                     << (trial_pack / INNER_REPS) << ","
                     << (trial_kernel / INNER_REPS) << ","
                     << ((trial_pack + trial_kernel) / INNER_REPS) << "\n";
        }

        // Calculate Medians
        sort(base_a_times.begin(), base_a_times.end());
        sort(base_b_times.begin(), base_b_times.end());
        sort(pack_times.begin(), pack_times.end());
        sort(kernel_times.begin(), kernel_times.end());
        sort(total_times.begin(), total_times.end());

        double median_base_a = base_a_times[MEASURED_TRIALS / 2];
        double median_base_b = base_b_times[MEASURED_TRIALS / 2];
        double median_pack = pack_times[MEASURED_TRIALS / 2];
        double median_kernel = kernel_times[MEASURED_TRIALS / 2];
        double median_total = total_times[MEASURED_TRIALS / 2];

        cout << "Results for " << shape.name << ":\n";
        cout << "  -> Median Baseline A Time: " << fixed << setprecision(2) << median_base_a << " us\n";
        cout << "  -> Median Baseline B Time: " << median_base_b << " us\n";
        cout << "  -> Median Pack Time:       " << median_pack << " us\n";
        cout << "  -> Median Kernel Time:     " << median_kernel << " us\n";
        cout << "  -> Median Total Time:      " << median_total << " us\n";
        cout << "--------------------------------------------------------\n";
    }
    
    csv_file.close();
    cout << "Data saved to benchmark_raw_arm64.csv\n";
}

int main() {
    run_benchmark();
    return 0;
}
