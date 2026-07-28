#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dav2 {

struct ImageShape {
    int width;
    int height;
};

ImageShape network_shape(int width, int height, int input_size);

void preprocess_bgr8(
    const std::uint8_t* source,
    int width,
    int height,
    std::ptrdiff_t stride,
    ImageShape destination,
    std::vector<float>& rgb_chw);

}  // namespace dav2
