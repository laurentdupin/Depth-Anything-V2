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

void resize_bilinear_align_corners(
    const float* source,
    int source_width,
    int source_height,
    float* destination,
    int destination_width,
    int destination_height);

}  // namespace dav2
