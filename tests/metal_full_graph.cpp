#include "depth_anything_v2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const char* model_path = std::getenv("DAV2_VITS_MODEL");
    if (model_path == nullptr || *model_path == '\0') {
        std::cout << "DAV2_VITS_MODEL is not set; skipping\n";
        return 77;
    }
    int width = 140;
    int height = 84;
    if (argc == 3) {
        width = std::atoi(argv[1]);
        height = std::atoi(argv[2]);
    } else if (argc != 1) {
        std::cerr << "usage: dav2_metal_full_graph [WIDTH HEIGHT]\n";
        return 64;
    }
    if (width <= 0 || height <= 0 || width % 14 != 0 || height % 14 != 0) {
        std::cerr << "dimensions must be positive multiples of 14\n";
        return 64;
    }
    std::vector<float> input(
        static_cast<std::size_t>(width) * height * 3u);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.001f);
    }
    std::vector<float> output(
        static_cast<std::size_t>(width) * height);
    const dav2_create_options options{
        sizeof(options), DAV2_ABI_VERSION, DAV2_ENCODER_VITS, 0,
        DAV2_CREATE_FORCE_METAL};
    dav2_context* context = nullptr;
    dav2_status status = dav2_create(model_path, &options, &context);
    if (status != DAV2_STATUS_OK) {
        std::cerr << "create: " << dav2_last_error() << '\n';
        return 1;
    }
    std::vector<double> samples;
    for (int iteration = 0; iteration < 6; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        status = dav2_infer_tensor_f32(
            context, input.data(), width, height,
            output.data(), output.size());
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (status != DAV2_STATUS_OK) {
            std::cerr << "infer: " << dav2_last_error() << '\n';
            dav2_destroy(context);
            return 2;
        }
        if (iteration != 0) samples.push_back(elapsed);
    }
    const auto bounds = std::minmax_element(output.begin(), output.end());
    if (!std::isfinite(*bounds.first) || !std::isfinite(*bounds.second) ||
        *bounds.second <= *bounds.first) {
        std::cerr << "invalid output range " << *bounds.first << " .. "
                  << *bounds.second << '\n';
        dav2_destroy(context);
        return 3;
    }
    std::sort(samples.begin(), samples.end());
    std::cout << "Metal full graph median: " << samples[samples.size() / 2]
              << " ms, range "
              << *bounds.first << " .. " << *bounds.second << '\n';
    dav2_destroy(context);
    return 0;
}
