#include "image.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dav2 {
namespace {

int round_to_multiple(double value, int multiple) {
    // numpy.round is ties-to-even. The dimensions encountered here are
    // positive, so nearbyint under the default rounding mode has the same rule.
    return static_cast<int>(std::nearbyint(value / multiple)) * multiple;
}

double cubic(double x) {
    // OpenCV INTER_CUBIC uses Keys' cubic convolution with a = -0.75.
    constexpr double a = -0.75;
    x = std::abs(x);
    if (x <= 1.0) {
        return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
    }
    if (x < 2.0) {
        return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
    }
    return 0.0;
}

int clamp_index(int value, int limit) {
    return std::max(0, std::min(value, limit - 1));
}

}  // namespace

ImageShape network_shape(int width, int height, int input_size) {
    if (width <= 0 || height <= 0 || input_size <= 0) {
        throw std::invalid_argument("image dimensions and input_size must be positive");
    }

    constexpr int patch_size = 14;
    const double scale = std::max(
        static_cast<double>(input_size) / static_cast<double>(height),
        static_cast<double>(input_size) / static_cast<double>(width));

    int output_height = round_to_multiple(scale * height, patch_size);
    int output_width = round_to_multiple(scale * width, patch_size);
    output_height = std::max(output_height, input_size);
    output_width = std::max(output_width, input_size);

    // Match Resize.constrain_to_multiple_of after its lower-bound correction.
    if (output_height % patch_size != 0) {
        output_height = ((output_height + patch_size - 1) / patch_size) * patch_size;
    }
    if (output_width % patch_size != 0) {
        output_width = ((output_width + patch_size - 1) / patch_size) * patch_size;
    }
    return {output_width, output_height};
}

void preprocess_bgr8(
    const std::uint8_t* source,
    int width,
    int height,
    std::ptrdiff_t stride,
    ImageShape destination,
    std::vector<float>& rgb_chw) {
    if (source == nullptr || width <= 0 || height <= 0 ||
        stride < static_cast<std::ptrdiff_t>(width) * 3 ||
        destination.width <= 0 || destination.height <= 0) {
        throw std::invalid_argument("invalid source image");
    }

    const std::size_t plane =
        static_cast<std::size_t>(destination.width) * destination.height;
    rgb_chw.resize(plane * 3);
    constexpr double mean[3] = {0.485, 0.456, 0.406};
    constexpr double stddev[3] = {0.229, 0.224, 0.225};

    const double scale_x = static_cast<double>(width) / destination.width;
    const double scale_y = static_cast<double>(height) / destination.height;
    for (int dy = 0; dy < destination.height; ++dy) {
        const double sy = (dy + 0.5) * scale_y - 0.5;
        const int iy = static_cast<int>(std::floor(sy));
        for (int dx = 0; dx < destination.width; ++dx) {
            const double sx = (dx + 0.5) * scale_x - 0.5;
            const int ix = static_cast<int>(std::floor(sx));
            double rgb[3] = {};
            for (int ky = -1; ky <= 2; ++ky) {
                const int py = clamp_index(iy + ky, height);
                const double wy = cubic(sy - (iy + ky));
                const std::uint8_t* row = source + static_cast<std::ptrdiff_t>(py) * stride;
                for (int kx = -1; kx <= 2; ++kx) {
                    const int px = clamp_index(ix + kx, width);
                    const double weight = wy * cubic(sx - (ix + kx));
                    const std::uint8_t* pixel = row + static_cast<std::ptrdiff_t>(px) * 3;
                    rgb[0] += weight * pixel[2];
                    rgb[1] += weight * pixel[1];
                    rgb[2] += weight * pixel[0];
                }
            }

            const std::size_t offset =
                static_cast<std::size_t>(dy) * destination.width + dx;
            for (int channel = 0; channel < 3; ++channel) {
                const double unit = rgb[channel] / 255.0;
                rgb_chw[static_cast<std::size_t>(channel) * plane + offset] =
                    static_cast<float>((unit - mean[channel]) / stddev[channel]);
            }
        }
    }
}

void resize_bilinear_align_corners(
    const float* source,
    int source_width,
    int source_height,
    float* destination,
    int destination_width,
    int destination_height) {
    if (source == nullptr || destination == nullptr || source_width <= 0 ||
        source_height <= 0 || destination_width <= 0 || destination_height <= 0) {
        throw std::invalid_argument("invalid resize image");
    }

    const double scale_x = destination_width > 1
        ? static_cast<double>(source_width - 1) / (destination_width - 1)
        : 0.0;
    const double scale_y = destination_height > 1
        ? static_cast<double>(source_height - 1) / (destination_height - 1)
        : 0.0;

    for (int dy = 0; dy < destination_height; ++dy) {
        const double sy = dy * scale_y;
        const int y0 = static_cast<int>(sy);
        const int y1 = std::min(y0 + 1, source_height - 1);
        const float fy = static_cast<float>(sy - y0);
        for (int dx = 0; dx < destination_width; ++dx) {
            const double sx = dx * scale_x;
            const int x0 = static_cast<int>(sx);
            const int x1 = std::min(x0 + 1, source_width - 1);
            const float fx = static_cast<float>(sx - x0);
            const float top =
                source[y0 * source_width + x0] * (1.0f - fx) +
                source[y0 * source_width + x1] * fx;
            const float bottom =
                source[y1 * source_width + x0] * (1.0f - fx) +
                source[y1 * source_width + x1] * fx;
            destination[dy * destination_width + dx] =
                top * (1.0f - fy) + bottom * fy;
        }
    }
}

}  // namespace dav2
