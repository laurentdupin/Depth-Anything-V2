#include "image.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace dav2 {
namespace {

int round_to_multiple(double value, int multiple) {
    // numpy.round is ties-to-even. The dimensions encountered here are
    // positive, so nearbyint under the default rounding mode has the same rule.
    return static_cast<int>(std::nearbyint(value / multiple)) * multiple;
}

void cubic_coefficients(float x, float coefficients[4]) {
    // Keep the float expressions and evaluation order used by OpenCV's
    // interpolateCubic. The Python reference resizes a float64 image, but
    // OpenCV deliberately computes interpolation coefficients in float32.
    constexpr float a = -0.75f;
    coefficients[0] =
        ((a * (x + 1.0f) - 5.0f * a) * (x + 1.0f) + 8.0f * a) *
            (x + 1.0f) -
        4.0f * a;
    coefficients[1] =
        ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    const float opposite = 1.0f - x;
    coefficients[2] =
        ((a + 2.0f) * opposite - (a + 3.0f)) *
            opposite * opposite +
        1.0f;
    coefficients[3] =
        1.0f - coefficients[0] - coefficients[1] - coefficients[2];
}

int clamp_index(int value, int limit) {
    return std::max(0, std::min(value, limit - 1));
}

template <typename Function>
void parallel_rows(int rows, Function&& function) {
    const unsigned available =
        std::max(1u, std::thread::hardware_concurrency());
    const unsigned workers = std::min<unsigned>(
        available, static_cast<unsigned>(rows));
    if (workers <= 1 || rows < 32) {
        function(0, rows);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(workers);
    try {
        for (unsigned worker = 0; worker < workers; ++worker) {
            const int begin =
                static_cast<int>(
                    std::uint64_t(rows) * worker / workers);
            const int end =
                static_cast<int>(
                    std::uint64_t(rows) * (worker + 1) / workers);
            threads.emplace_back([&, begin, end] {
                function(begin, end);
            });
        }
    } catch (...) {
        for (std::thread& thread : threads) {
            thread.join();
        }
        throw;
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
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
    ImageScratch& scratch,
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
    scratch.x_indices.resize(static_cast<std::size_t>(destination.width));
    scratch.x_coefficients.resize(
        static_cast<std::size_t>(destination.width) * 4);
    for (int dx = 0; dx < destination.width; ++dx) {
        const float coordinate =
            static_cast<float>((dx + 0.5) * scale_x - 0.5);
        const int index = static_cast<int>(std::floor(coordinate));
        scratch.x_indices[dx] = index;
        cubic_coefficients(
            coordinate - index,
            scratch.x_coefficients.data() +
                static_cast<std::size_t>(dx) * 4);
    }

    // OpenCV performs cubic resize as a separable filter with double
    // intermediate rows and float coefficients for a CV_64F input.
    scratch.horizontal.resize(
        static_cast<std::size_t>(height) * destination.width * 3);
    parallel_rows(height, [&](int begin, int end) {
        for (int sy = begin; sy < end; ++sy) {
            const std::uint8_t* row =
                source + static_cast<std::ptrdiff_t>(sy) * stride;
            for (int dx = 0; dx < destination.width; ++dx) {
                const float* alpha =
                    scratch.x_coefficients.data() +
                    static_cast<std::size_t>(dx) * 4;
                const int base = scratch.x_indices[dx];
                for (int channel = 0; channel < 3; ++channel) {
                    double value = 0.0;
                    for (int tap = 0; tap < 4; ++tap) {
                        const int sx = clamp_index(base - 1 + tap, width);
                        const std::uint8_t* pixel =
                            row + static_cast<std::ptrdiff_t>(sx) * 3;
                        const int bgr_channel = 2 - channel;
                        const double unit =
                            static_cast<double>(pixel[bgr_channel]) / 255.0;
                        value += unit * alpha[tap];
                    }
                    scratch.horizontal[
                        (static_cast<std::size_t>(sy) *
                            destination.width + dx) *
                            3 +
                        channel] = value;
                }
            }
        }
    });

    parallel_rows(destination.height, [&](int begin, int end) {
        for (int dy = begin; dy < end; ++dy) {
            const float coordinate =
                static_cast<float>((dy + 0.5) * scale_y - 0.5);
            const int base = static_cast<int>(std::floor(coordinate));
            float beta[4];
            cubic_coefficients(coordinate - base, beta);
            for (int dx = 0; dx < destination.width; ++dx) {
                const std::size_t offset =
                    static_cast<std::size_t>(dy) * destination.width + dx;
                for (int channel = 0; channel < 3; ++channel) {
                    const auto sample = [&](int tap) {
                        const int sy =
                            clamp_index(base - 1 + tap, height);
                        return scratch.horizontal[
                            (static_cast<std::size_t>(sy) *
                                destination.width + dx) *
                                3 +
                            channel];
                    };
                    const double resized =
                        sample(0) * beta[0] + sample(1) * beta[1] +
                        sample(2) * beta[2] + sample(3) * beta[3];
                    rgb_chw[
                        static_cast<std::size_t>(channel) * plane + offset] =
                        static_cast<float>(
                            (resized - mean[channel]) / stddev[channel]);
                }
            }
        }
    });
}

void preprocess_bgr8(
    const std::uint8_t* source,
    int width,
    int height,
    std::ptrdiff_t stride,
    ImageShape destination,
    std::vector<float>& rgb_chw) {
    ImageScratch scratch;
    preprocess_bgr8(
        source,
        width,
        height,
        stride,
        destination,
        scratch,
        rgb_chw);
}

}  // namespace dav2
