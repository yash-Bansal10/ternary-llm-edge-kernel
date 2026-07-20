#include <torch/extension.h>
#include <vector>
#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#define POPCOUNT __popcnt
#else
#define POPCOUNT __builtin_popcount
#endif

// The Bit-Serial MAC optimized kernel wrapped for PyTorch
int32_t optimized_bit_serial_mac(
    torch::Tensor pos_plane,
    torch::Tensor neg_plane,
    torch::Tensor bit_acts) 
{
    // Ensure 1D contiguous tensors for simplicity
    auto pos_data = pos_plane.contiguous().data_ptr<int32_t>();
    auto neg_data = neg_plane.contiguous().data_ptr<int32_t>();
    auto acts_data = bit_acts.contiguous().data_ptr<int32_t>();

    int num_words = pos_plane.size(0);
    int32_t accumulator = 0;

    // 8 bit-planes
    for (int b = 0; b < 8; ++b) {
        int32_t plane_sum = 0;
        
        for (int i = 0; i < num_words; ++i) {
            uint32_t pos = static_cast<uint32_t>(pos_data[i]);
            uint32_t neg = static_cast<uint32_t>(neg_data[i]);
            uint32_t act_bits = static_cast<uint32_t>(acts_data[b * num_words + i]);

            uint32_t pos_match = pos & act_bits;
            uint32_t neg_match = neg & act_bits;

            plane_sum += POPCOUNT(pos_match);
            plane_sum -= POPCOUNT(neg_match);
        }

        if (b == 7) {
            accumulator -= (plane_sum << b);
        } else {
            accumulator += (plane_sum << b);
        }
    }
    return accumulator;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_bit_serial_mac, "Ternary Bit-Serial MAC forward");
}
