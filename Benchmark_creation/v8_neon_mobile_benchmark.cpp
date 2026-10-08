


#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <arm_neon.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#define POPCOUNT32 __popcnt
#define NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define POPCOUNT32 __builtin_popcount
#define NOINLINE __attribute__((noinline))
#else
#define POPCOUNT32 __builtin_popcount
#define NOINLINE
#endif

volatile int64_t benchmark_sink = 0;

// ============================================================
// Packed ternary layer
// ============================================================

struct PackedLayer {
    uint32_t rows;
    uint32_t words_per_row;
    uint32_t original_cols;

    std::vector<uint32_t> pos;
    std::vector<uint32_t> neg;
};

// ============================================================
// Load packed layer
// ============================================================

PackedLayer load_layer(const std::string& filename) {

    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "Could not open: " + filename
        );
    }

    PackedLayer layer;

    file.read(
        reinterpret_cast<char*>(&layer.rows),
        sizeof(uint32_t)
    );

    file.read(
        reinterpret_cast<char*>(&layer.words_per_row),
        sizeof(uint32_t)
    );

    file.read(
        reinterpret_cast<char*>(&layer.original_cols),
        sizeof(uint32_t)
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to read header."
        );
    }

    const size_t total_words =
        static_cast<size_t>(layer.rows)
        * layer.words_per_row;

    layer.pos.resize(total_words);
    layer.neg.resize(total_words);

    file.read(
        reinterpret_cast<char*>(layer.pos.data()),
        total_words * sizeof(uint32_t)
    );

    file.read(
        reinterpret_cast<char*>(layer.neg.data()),
        total_words * sizeof(uint32_t)
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to read packed weight data."
        );
    }

    return layer;
}

// ============================================================
// Reconstruct dense INT8 ternary weights
// ============================================================

std::vector<int8_t> reconstruct_dense_weights(
    const PackedLayer& layer
) {

    const size_t total_elements =
        static_cast<size_t>(layer.rows)
        * layer.original_cols;

    std::vector<int8_t> dense(total_elements);

    for (uint32_t r = 0; r < layer.rows; ++r) {

        const size_t packed_offset =
            static_cast<size_t>(r)
            * layer.words_per_row;

        const size_t dense_offset =
            static_cast<size_t>(r)
            * layer.original_cols;

        for (uint32_t c = 0;
             c < layer.original_cols;
             ++c) {

            const uint32_t word = c / 32;
            const uint32_t bit = c % 32;

            const uint32_t p =
                layer.pos[packed_offset + word];

            const uint32_t n =
                layer.neg[packed_offset + word];

            const int32_t weight =
                static_cast<int32_t>(
                    (p >> bit) & 1U
                )
                -
                static_cast<int32_t>(
                    (n >> bit) & 1U
                );

            dense[dense_offset + c] =
                static_cast<int8_t>(weight);
        }
    }

    return dense;
}

// ============================================================
// BASELINE A
//
// Dense INT8 ternary weights.
// Scalar multiply-accumulate.
// ============================================================

NOINLINE int64_t baseline_a_dense(
    const std::vector<int8_t>& weights,
    const std::vector<int8_t>& activations,
    uint32_t rows,
    uint32_t cols
) {

    int64_t checksum = 0;

    for (uint32_t r = 0; r < rows; ++r) {

        int32_t acc = 0;

        const size_t offset =
            static_cast<size_t>(r) * cols;

        for (uint32_t c = 0; c < cols; ++c) {

            acc +=
                static_cast<int32_t>(
                    weights[offset + c]
                )
                *
                static_cast<int32_t>(
                    activations[c]
                );
        }

        checksum += acc;
    }

    return checksum;
}

// ============================================================
// BASELINE B
//
// Packed ternary weights.
// Scalar on-the-fly decoding.
// ============================================================

NOINLINE int64_t baseline_b_unpack(
    const PackedLayer& layer,
    const std::vector<int8_t>& activations
) {

    int64_t checksum = 0;

    for (uint32_t r = 0;
         r < layer.rows;
         ++r) {

        int32_t acc = 0;

        const size_t offset =
            static_cast<size_t>(r)
            * layer.words_per_row;

        for (uint32_t c = 0;
             c < layer.original_cols;
             ++c) {

            const uint32_t word = c / 32;
            const uint32_t bit = c % 32;

            const uint32_t p =
                layer.pos[offset + word];

            const uint32_t n =
                layer.neg[offset + word];

            const int32_t weight =
                static_cast<int32_t>(
                    (p >> bit) & 1U
                )
                -
                static_cast<int32_t>(
                    (n >> bit) & 1U
                );

            acc +=
                weight
                *
                static_cast<int32_t>(
                    activations[c]
                );
        }

        checksum += acc;
    }

    return checksum;
}

// ============================================================
// Allocate activation bitplanes
// ============================================================

void allocate_activation_planes(
    std::array<std::vector<uint32_t>, 8>& planes,
    uint32_t words_per_row
) {

    for (auto& plane : planes) {
        plane.resize(words_per_row);
    }
}

// ============================================================
// Activation packing
//
// INT8 -> 8 bitplanes
// ============================================================

NOINLINE void pack_activations_reuse(
    const std::vector<int8_t>& activations,
    std::array<std::vector<uint32_t>, 8>& planes
) {

    for (auto& plane : planes) {
        std::fill(
            plane.begin(),
            plane.end(),
            0U
        );
    }

    for (uint32_t c = 0;
         c < activations.size();
         ++c) {

        const uint8_t value =
            static_cast<uint8_t>(
                activations[c]
            );

        const uint32_t word = c / 32;
        const uint32_t bit = c % 32;

        const uint32_t mask =
            1U << bit;

        for (int b = 0; b < 8; ++b) {

            if ((value >> b) & 1U) {
                planes[b][word] |= mask;
            }
        }
    }
}

// ============================================================
// C: Scalar bit-serial reference kernel
// ============================================================

NOINLINE int64_t proposed_bitserial_scalar(
    const PackedLayer& layer,
    const std::array<std::vector<uint32_t>, 8>& planes
) {

    int64_t checksum = 0;

    for (uint32_t r = 0;
         r < layer.rows;
         ++r) {

        int32_t acc = 0;

        const size_t offset =
            static_cast<size_t>(r)
            * layer.words_per_row;

        for (int b = 0; b < 8; ++b) {

            int32_t plane_acc = 0;

            const int32_t scale =
                (b == 7)
                ? -128
                : (1 << b);

            for (uint32_t w = 0;
                 w < layer.words_per_row;
                 ++w) {

                const uint32_t activation_bits =
                    planes[b][w];

                plane_acc +=
                    POPCOUNT32(
                        layer.pos[offset + w]
                        & activation_bits
                    );

                plane_acc -=
                    POPCOUNT32(
                        layer.neg[offset + w]
                        & activation_bits
                    );
            }

            acc +=
                plane_acc * scale;
        }

        checksum += acc;
    }

    return checksum;
}

// ============================================================
// D: NEON bit-serial kernel (Partial 128-bit)
// ============================================================
__attribute__((noinline)) int64_t proposed_bitserial_neon_partial(
    const PackedLayer& layer,
    const std::array<std::vector<uint32_t>, 8>& planes
) {
    int64_t checksum = 0;
    for (uint32_t r = 0; r < layer.rows; ++r) {
        int32_t acc = 0;
        const size_t offset = static_cast<size_t>(r) * layer.words_per_row;
        for (int b = 0; b < 8; ++b) {
            int32_t plane_acc = 0;
            const int32_t scale = (b == 7) ? -128 : (1 << b);
            uint32_t w = 0;
            
            for (; w + 4 <= layer.words_per_row; w += 4) {
                uint32x4_t pos_vec = vld1q_u32(layer.pos.data() + offset + w);
                uint32x4_t neg_vec = vld1q_u32(layer.neg.data() + offset + w);
                uint32x4_t act_vec = vld1q_u32(planes[b].data() + w);

                uint32x4_t pos_and = vandq_u32(pos_vec, act_vec);
                uint32x4_t neg_and = vandq_u32(neg_vec, act_vec);

                uint32_t p_lanes[4];
                uint32_t n_lanes[4];
                vst1q_u32(p_lanes, pos_and);
                vst1q_u32(n_lanes, neg_and);

                plane_acc += POPCOUNT32(p_lanes[0]) + POPCOUNT32(p_lanes[1]) + POPCOUNT32(p_lanes[2]) + POPCOUNT32(p_lanes[3]);
                plane_acc -= (POPCOUNT32(n_lanes[0]) + POPCOUNT32(n_lanes[1]) + POPCOUNT32(n_lanes[2]) + POPCOUNT32(n_lanes[3]));
            }

            for (; w < layer.words_per_row; ++w) {
                const uint32_t activation_bits = planes[b][w];
                plane_acc += POPCOUNT32(layer.pos[offset + w] & activation_bits);
                plane_acc -= POPCOUNT32(layer.neg[offset + w] & activation_bits);
            }
            acc += plane_acc * scale;
        }
        checksum += acc;
    }
    return checksum;
}

// ============================================================
// E: Fully vectorized bit-serial kernel (NEON)
// ============================================================
__attribute__((noinline)) int64_t proposed_bitserial_neon_full(
    const PackedLayer& layer,
    const std::array<std::vector<uint32_t>, 8>& planes
) {
    int64_t checksum = 0;
    for (uint32_t r = 0; r < layer.rows; ++r) {
        int32_t acc = 0;
        const size_t offset = static_cast<size_t>(r) * layer.words_per_row;
        for (int b = 0; b < 8; ++b) {
            int32_t plane_acc = 0;
            const int32_t scale = (b == 7) ? -128 : (1 << b);
            uint32_t w = 0;

            for (; w + 4 <= layer.words_per_row; w += 4) {
                uint32x4_t pos_vec = vld1q_u32(layer.pos.data() + offset + w);
                uint32x4_t neg_vec = vld1q_u32(layer.neg.data() + offset + w);
                uint32x4_t act_vec = vld1q_u32(planes[b].data() + w);

                uint32x4_t pos_and = vandq_u32(pos_vec, act_vec);
                uint32x4_t neg_and = vandq_u32(neg_vec, act_vec);

                uint8x16_t pos_8 = vreinterpretq_u8_u32(pos_and);
                uint8x16_t neg_8 = vreinterpretq_u8_u32(neg_and);

                uint8x16_t pos_cnt = vcntq_u8(pos_8);
                uint8x16_t neg_cnt = vcntq_u8(neg_8);

                uint16x8_t pos_sum16 = vpaddlq_u8(pos_cnt);
                uint16x8_t neg_sum16 = vpaddlq_u8(neg_cnt);

                uint32x4_t pos_sum32 = vpaddlq_u16(pos_sum16);
                uint32x4_t neg_sum32 = vpaddlq_u16(neg_sum16);

                uint32_t p_cnt = vaddvq_u32(pos_sum32);
                uint32_t n_cnt = vaddvq_u32(neg_sum32);

                plane_acc += (p_cnt - n_cnt);
            }

            for (; w < layer.words_per_row; ++w) {
                const uint32_t activation_bits = planes[b][w];
                plane_acc += POPCOUNT32(layer.pos[offset + w] & activation_bits);
                plane_acc -= POPCOUNT32(layer.neg[offset + w] & activation_bits);
            }
            acc += plane_acc * scale;
        }
        checksum += acc;
    }
    return checksum;
}


// ============================================================

// Activation index
// ============================================================

inline int activation_index(
    int trial,
    int rep,
    int inner_reps,
    int activation_sets
) {

    return (
        trial * inner_reps + rep
    ) % activation_sets;
}

// ============================================================
// Time A
// ============================================================

double time_baseline_a_us(
    const std::vector<int8_t>& dense_weights,
    const std::vector<std::vector<int8_t>>& activation_pool,
    uint32_t rows,
    uint32_t cols,
    int trial,
    int repetitions
) {

    int64_t local_checksum = 0;

    const int activation_sets =
        static_cast<int>(
            activation_pool.size()
        );

    const auto start =
        std::chrono::steady_clock::now();

    for (int rep = 0;
         rep < repetitions;
         ++rep) {

        const int idx =
            activation_index(
                trial,
                rep,
                repetitions,
                activation_sets
            );

        local_checksum +=
            baseline_a_dense(
                dense_weights,
                activation_pool[idx],
                rows,
                cols
            );
    }

    const auto end =
        std::chrono::steady_clock::now();

    benchmark_sink += local_checksum;

    return std::chrono::duration<
        double,
        std::micro
    >(end - start).count() / repetitions;
}

// ============================================================
// Time B
// ============================================================

double time_baseline_b_us(
    const PackedLayer& layer,
    const std::vector<std::vector<int8_t>>& activation_pool,
    int trial,
    int repetitions
) {

    int64_t local_checksum = 0;

    const int activation_sets =
        static_cast<int>(
            activation_pool.size()
        );

    const auto start =
        std::chrono::steady_clock::now();

    for (int rep = 0;
         rep < repetitions;
         ++rep) {

        const int idx =
            activation_index(
                trial,
                rep,
                repetitions,
                activation_sets
            );

        local_checksum +=
            baseline_b_unpack(
                layer,
                activation_pool[idx]
            );
    }

    const auto end =
        std::chrono::steady_clock::now();

    benchmark_sink += local_checksum;

    return std::chrono::duration<
        double,
        std::micro
    >(end - start).count() / repetitions;
}

// ============================================================
// Generic prepacked kernel timer
//
// Used for both C and D.
// ============================================================

template <typename Kernel>
double time_prepacked_kernel_us(
    Kernel kernel,
    const PackedLayer& layer,
    const std::vector<
        std::array<
            std::vector<uint32_t>,
            8
        >
    >& prepacked_planes,
    int trial,
    int repetitions
) {

    int64_t local_checksum = 0;

    const int activation_sets =
        static_cast<int>(
            prepacked_planes.size()
        );

    const auto start =
        std::chrono::steady_clock::now();

    for (int rep = 0;
         rep < repetitions;
         ++rep) {

        const int idx =
            activation_index(
                trial,
                rep,
                repetitions,
                activation_sets
            );

        local_checksum +=
            kernel(
                layer,
                prepacked_planes[idx]
            );
    }

    const auto end =
        std::chrono::steady_clock::now();

    benchmark_sink += local_checksum;

    return std::chrono::duration<
        double,
        std::micro
    >(end - start).count() / repetitions;
}

// ============================================================
// Time packing
// ============================================================

double time_pack_us(
    const std::vector<std::vector<int8_t>>& activation_pool,
    std::array<std::vector<uint32_t>, 8>& planes,
    int trial,
    int repetitions
) {

    const int activation_sets =
        static_cast<int>(
            activation_pool.size()
        );

    uint64_t local_touch = 0;

    const auto start =
        std::chrono::steady_clock::now();

    for (int rep = 0;
         rep < repetitions;
         ++rep) {

        const int idx =
            activation_index(
                trial,
                rep,
                repetitions,
                activation_sets
            );

        pack_activations_reuse(
            activation_pool[idx],
            planes
        );

        local_touch +=
            planes[0][0];
    }

    const auto end =
        std::chrono::steady_clock::now();

    benchmark_sink +=
        static_cast<int64_t>(
            local_touch
        );

    return std::chrono::duration<
        double,
        std::micro
    >(end - start).count() / repetitions;
}

// ============================================================
// Direct D E2E
//
// pack activation + AVX2 kernel
// ============================================================

double time_neon_partial_e2e_us(
    const PackedLayer& layer,
    const std::vector<std::vector<int8_t>>& activation_pool,
    std::array<std::vector<uint32_t>, 8>& planes,
    int trial,
    int repetitions
) {

    int64_t local_checksum = 0;

    const int activation_sets =
        static_cast<int>(
            activation_pool.size()
        );

    const auto start =
        std::chrono::steady_clock::now();

    for (int rep = 0;
         rep < repetitions;
         ++rep) {

        const int idx =
            activation_index(
                trial,
                rep,
                repetitions,
                activation_sets
            );

        pack_activations_reuse(
            activation_pool[idx],
            planes
        );

        local_checksum +=
            proposed_bitserial_neon_partial(
                layer,
                planes
            );
    }

    const auto end =
        std::chrono::steady_clock::now();

    benchmark_sink += local_checksum;

    return std::chrono::duration<
        double,
        std::micro
    >(end - start).count() / repetitions;
}

// ============================================================
// Direct E E2E timing
//
// Includes activation packing + fully vectorized V7 kernel.
// ============================================================

double time_neon_full_e2e_us(
    const PackedLayer& layer,
    const std::vector<std::vector<int8_t>>& activation_pool,
    std::array<std::vector<uint32_t>, 8>& planes,
    int trial,
    int repetitions
) {

    int64_t local_checksum = 0;

    const int activation_sets =
        static_cast<int>(activation_pool.size());

    const auto start =
        std::chrono::steady_clock::now();

    for (int rep = 0; rep < repetitions; ++rep) {

        const int idx =
            activation_index(
                trial,
                rep,
                repetitions,
                activation_sets
            );

        pack_activations_reuse(
            activation_pool[idx],
            planes
        );

        local_checksum +=
            proposed_bitserial_neon_full(
                layer,
                planes
            );
    }

    const auto end =
        std::chrono::steady_clock::now();

    benchmark_sink += local_checksum;

    return std::chrono::duration<double, std::micro>(
        end - start
    ).count() / repetitions;
}

// ============================================================
// Main
// ============================================================

int main(
    int argc,
    char* argv[]
) {

    if (argc != 4) {

        std::cerr
            << "Usage:\n"
            << "./benchmark_v7 "
            << "<layer.bin> "
            << "<layer_name> "
            << "<output.csv>\n";

        return 1;
    }

    const std::string binary_path =
        argv[1];

    const std::string layer_name =
        argv[2];

    const std::string csv_path =
        argv[3];

    constexpr int WARMUP_ROUNDS = 30;
    constexpr int TRIALS = 60;
    constexpr int INNER_REPS = 10;
    constexpr int ACTIVATION_SETS = 64;

    // ========================================================
    // Balanced C/D/E execution order
    //
    // A and B remain reference baselines.
    // C, D, and E are the kernel comparison set.
    // ========================================================

    // Six permutations of C/D/E.  Each appears exactly 10 times
    // over 60 trials, then the trial sequence is deterministically
    // shuffled to reduce systematic order effects.
    const std::array<std::array<int, 3>, 6> permutations = {{
        {{2, 3, 4}}, // C-D-E
        {{2, 4, 3}}, // C-E-D
        {{3, 2, 4}}, // D-C-E
        {{3, 4, 2}}, // D-E-C
        {{4, 2, 3}}, // E-C-D
        {{4, 3, 2}}  // E-D-C
    }};

    std::vector<int> order_ids(TRIALS);
    for (int i = 0; i < TRIALS; ++i) {
        order_ids[i] = i % 6;
    }

    std::mt19937 order_rng(2026);
    std::shuffle(order_ids.begin(), order_ids.end(), order_rng);

    // ========================================================
    // Load
    // ========================================================

    PackedLayer layer =
        load_layer(
            binary_path
        );

    std::cout
        << "\n========================================\n"
        << "FINAL BENCHMARK V7 - NEON SIMD POPCOUNT\n"
        << "========================================\n"
        << "Layer           : "
        << layer_name
        << "\n"
        << "Matrix          : "
        << layer.rows
        << " x "
        << layer.original_cols
        << "\n"
        << "Words/row       : "
        << layer.words_per_row
        << "\n"
        << "Warmups         : "
        << WARMUP_ROUNDS
        << "\n"
        << "Trials          : "
        << TRIALS
        << "\n"
        << "Inner reps      : "
        << INNER_REPS
        << "\n"
        << "Activation sets : "
        << ACTIVATION_SETS
        << "\n"
        << "SIMD            : ARM NEON + Native vector popcount\n"
        << "========================================\n";

    // ========================================================
    // Dense weights
    // ========================================================

    std::cout
        << "[1] Reconstructing dense INT8 weights...\n";

    std::vector<int8_t> dense_weights =
        reconstruct_dense_weights(
            layer
        );

    // ========================================================
    // Activation pool
    // ========================================================

    std::cout
        << "[2] Generating activation pool...\n";

    std::vector<
        std::vector<int8_t>
    > activation_pool(

        ACTIVATION_SETS,

        std::vector<int8_t>(
            layer.original_cols
        )
    );

    std::mt19937 activation_rng(42);

    std::uniform_int_distribution<int>
        distribution(-128, 127);

    for (auto& activations :
         activation_pool) {

        for (auto& value :
             activations) {

            value =
                static_cast<int8_t>(
                    distribution(
                        activation_rng
                    )
                );
        }
    }

    // ========================================================
    // Prepack activation pool
    // ========================================================

    std::cout
        << "[3] Prepacking activation pool...\n";

    std::vector<
        std::array<
            std::vector<uint32_t>,
            8
        >
    > prepacked_planes(
        ACTIVATION_SETS
    );

    for (int i = 0;
         i < ACTIVATION_SETS;
         ++i) {

        allocate_activation_planes(
            prepacked_planes[i],
            layer.words_per_row
        );

        pack_activations_reuse(
            activation_pool[i],
            prepacked_planes[i]
        );
    }

    std::array<
        std::vector<uint32_t>,
        8
    > pack_planes;

    std::array<
        std::vector<uint32_t>,
        8
    > e2e_planes;

    allocate_activation_planes(
        pack_planes,
        layer.words_per_row
    );

    allocate_activation_planes(
        e2e_planes,
        layer.words_per_row
    );

    // ========================================================
    // Four-way correctness
    // ========================================================

    std::cout
        << "[4] Running 64-input A/B/C/D/E correctness check...\n";

    int64_t aggregate_a = 0;
    int64_t aggregate_b = 0;
    int64_t aggregate_c = 0;
    int64_t aggregate_d = 0;
    int64_t aggregate_e = 0;

    for (int i = 0;
         i < ACTIVATION_SETS;
         ++i) {

        const int64_t a =
            baseline_a_dense(
                dense_weights,
                activation_pool[i],
                layer.rows,
                layer.original_cols
            );

        const int64_t b =
            baseline_b_unpack(
                layer,
                activation_pool[i]
            );

        const int64_t c =
            proposed_bitserial_scalar(
                layer,
                prepacked_planes[i]
            );

        const int64_t d =
            proposed_bitserial_neon_partial(
                layer,
                prepacked_planes[i]
            );

        const int64_t e =
            proposed_bitserial_neon_full(
                layer,
                prepacked_planes[i]
            );

        if (
            a != b
            ||
            a != c
            ||
            a != d
            ||
            a != e
        ) {

            std::cerr
                << "\nFAIL at activation set "
                << i
                << "\n"
                << "A = "
                << a
                << "\n"
                << "B = "
                << b
                << "\n"
                << "C = "
                << c
                << "\n"
                << "D = "
                << d
                << "\n"
                << "E = "
                << e
                << "\n";

            return 2;
        }

        aggregate_a += a;
        aggregate_b += b;
        aggregate_c += c;
        aggregate_d += d;
        aggregate_e += e;
    }

    std::cout
        << "    PASS: A == B == C == D for all "
        << ACTIVATION_SETS
        << " activation sets\n"
        << "    Aggregate A = "
        << aggregate_a
        << "\n"
        << "    Aggregate B = "
        << aggregate_b
        << "\n"
        << "    Aggregate C = "
        << aggregate_c
        << "\n"
        << "    Aggregate D = "
        << aggregate_d
        << "\n"
        << "    Aggregate E = "
        << aggregate_e
        << "\n";

    // ========================================================
    // Warmup
    // ========================================================

    std::cout
        << "[5] Running warm-up rounds...\n";

    for (int i = 0;
         i < WARMUP_ROUNDS;
         ++i) {

        const int idx =
            i % ACTIVATION_SETS;

        benchmark_sink +=
            baseline_a_dense(
                dense_weights,
                activation_pool[idx],
                layer.rows,
                layer.original_cols
            );

        benchmark_sink +=
            baseline_b_unpack(
                layer,
                activation_pool[idx]
            );

        benchmark_sink +=
            proposed_bitserial_scalar(
                layer,
                prepacked_planes[idx]
            );

        benchmark_sink +=
            proposed_bitserial_neon_partial(
                layer,
                prepacked_planes[idx]
            );

        benchmark_sink +=
            proposed_bitserial_neon_full(
                layer,
                prepacked_planes[idx]
            );

        pack_activations_reuse(
            activation_pool[idx],
            e2e_planes
        );

        benchmark_sink +=
            proposed_bitserial_neon_full(
                layer,
                e2e_planes
            );
    }

    // ========================================================
    // CSV
    // ========================================================

    const bool csv_exists =
        std::ifstream(
            csv_path
        ).good();

    std::ofstream csv(
        csv_path,
        std::ios::app
    );

    if (!csv) {

        std::cerr
            << "Could not open CSV: "
            << csv_path
            << "\n";

        return 3;
    }

    if (!csv_exists) {

        csv
            << "layer,"
            << "trial,"
            << "cd_order,"
            << "baseline_a_us,"
            << "baseline_b_us,"
            << "activation_pack_us,"
            << "scalar_c_kernel_us,"
            << "neon_d_kernel_us,"
            << "neon_e_kernel_us,"
            << "lut_component_sum_us,"
            << "lut_direct_e2e_us,"
            << "e_vs_scalar_speedup,"
            << "e_vs_partial_avx2_speedup,"
            << "dense_vs_e_kernel_ratio,"
            << "dense_vs_e_e2e_ratio,"
            << "e2e_to_component_ratio\n";
    }

    // ========================================================
    // Trials
    // ========================================================

    std::cout
        << "[6] Starting measured trials...\n";

    for (int trial = 0;
         trial < TRIALS;
         ++trial) {

        const double time_a =
            time_baseline_a_us(
                dense_weights,
                activation_pool,
                layer.rows,
                layer.original_cols,
                trial,
                INNER_REPS
            );

        const double time_b =
            time_baseline_b_us(
                layer,
                activation_pool,
                trial,
                INNER_REPS
            );

        const double activation_pack_us =
            time_pack_us(
                activation_pool,
                pack_planes,
                trial,
                INNER_REPS
            );

        double times[3] = {0.0, 0.0, 0.0};

        std::string cde_order;
        const auto& order = permutations[order_ids[trial]];

        for (int slot = 0; slot < 3; ++slot) {

            const int kernel_id = order[slot];

            switch (kernel_id) {

                case 2:
                    times[0] = time_prepacked_kernel_us(
                        proposed_bitserial_scalar,
                        layer,
                        prepacked_planes,
                        trial,
                        INNER_REPS
                    );
                    cde_order += (slot ? "-C" : "C");
                    break;

                case 3:
                    times[1] = time_prepacked_kernel_us(
                        proposed_bitserial_neon_partial,
                        layer,
                        prepacked_planes,
                        trial,
                        INNER_REPS
                    );
                    cde_order += (slot ? "-D" : "D");
                    break;

                case 4:
                    times[2] = time_prepacked_kernel_us(
                        proposed_bitserial_neon_full,
                        layer,
                        prepacked_planes,
                        trial,
                        INNER_REPS
                    );
                    cde_order += (slot ? "-E" : "E");
                    break;

                default:
                    throw std::runtime_error("Invalid C/D/E order id.");
            }
        }

        const double time_c = times[0];
        const double time_d = times[1];
        const double time_e = times[2];

        const double e_component_sum_us =
            activation_pack_us + time_e;

        const double e_direct_e2e_us =
            time_neon_full_e2e_us(
                layer,
                activation_pool,
                e2e_planes,
                trial,
                INNER_REPS
            );

        const double e_vs_scalar_speedup =
            time_e > 0.0 ? time_c / time_e : 0.0;

        const double e_vs_partial_avx2_speedup =
            time_e > 0.0 ? time_d / time_e : 0.0;

        const double dense_vs_e_kernel_ratio =
            time_e > 0.0 ? time_a / time_e : 0.0;

        const double dense_vs_e_e2e_ratio =
            e_direct_e2e_us > 0.0
                ? time_a / e_direct_e2e_us
                : 0.0;

        const double e_e2e_to_component_ratio =
            e_component_sum_us > 0.0
                ? e_direct_e2e_us / e_component_sum_us
                : 0.0;

        csv
            << layer_name
            << ","
            << trial
            << ","
            << cde_order
            << ","
            << std::fixed
            << std::setprecision(3)
            << time_a
            << ","
            << time_b
            << ","
            << activation_pack_us
            << ","
            << time_c
            << ","
            << time_d
            << ","
            << time_e
            << ","
            << e_component_sum_us
            << ","
            << e_direct_e2e_us
            << ","
            << std::setprecision(6)
            << e_vs_scalar_speedup
            << ","
            << e_vs_partial_avx2_speedup
            << ","
            << dense_vs_e_kernel_ratio
            << ","
            << dense_vs_e_e2e_ratio
            << ","
            << e_e2e_to_component_ratio
            << "\n";

        std::cout
            << "Trial "
            << std::setw(2)
            << trial
            << " ["
            << cde_order
            << "] "
            << std::fixed
            << std::setprecision(2)
            << "A=" << time_a << " us  "
            << "B=" << time_b << " us  "
            << "C=" << time_c << " us  "
            << "D=" << time_d << " us  "
            << "E=" << time_e << " us  "
            << "Pack=" << activation_pack_us << " us  "
            << "E-E2E=" << e_direct_e2e_us << " us  "
            << "C/E=" << std::setprecision(3)
            << e_vs_scalar_speedup << "x  "
            << "D/E="
            << e_vs_partial_avx2_speedup << "x  "
            << "A/E="
            << dense_vs_e_kernel_ratio << "x  "
            << "A/E-E2E="
            << dense_vs_e_e2e_ratio << "x\n";
    }

    csv.close();

    std::cout
        << "\n========================================\n"
        << "BENCHMARK V7 COMPLETE\n"
        << "========================================\n"
        << "Layer: "
        << layer_name
        << "\n"
        << "Checksum sink: "
        << benchmark_sink
        << "\n"
        << "CSV: "
        << csv_path
        << "\n";

    return 0;
}