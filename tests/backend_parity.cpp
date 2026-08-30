#include "depth_anything_v2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

struct RunResult {
    std::vector<float> depth;
    double milliseconds = 0.0;
};

bool run(
    const char* model_path, std::uint32_t backend_flag,
    const std::vector<float>& input, int width, int height,
    RunResult& result) {
    const dav2_create_options options{
        sizeof(options), DAV2_ABI_VERSION, DAV2_ENCODER_VITS, 0,
        backend_flag | DAV2_CREATE_FORCE_FP32};
    dav2_context* context = nullptr;
    dav2_status status = dav2_create(model_path, &options, &context);
    if (status != DAV2_STATUS_OK) {
        std::cerr << "create failed: " << dav2_last_error() << '\n';
        return false;
    }
    result.depth.resize(static_cast<std::size_t>(width) * height);
    // The first call includes shader/graph compilation and backend tuning.
    status = dav2_infer_tensor_f32(
        context, input.data(), width, height,
        result.depth.data(), result.depth.size());
    if (status == DAV2_STATUS_OK) {
        const auto start = std::chrono::steady_clock::now();
        status = dav2_infer_tensor_f32(
            context, input.data(), width, height,
            result.depth.data(), result.depth.size());
        result.milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }
    if (status != DAV2_STATUS_OK) {
        std::cerr << "infer failed: " << dav2_last_error() << '\n';
        dav2_destroy(context);
        return false;
    }
    dav2_destroy(context);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model_path = std::getenv("DAV2_VITS_MODEL");
    if (model_path == nullptr || *model_path == '\0') {
        std::cout << "DAV2_VITS_MODEL is not set; skipping\n";
        return 77;
    }
    int width = 280;
    int height = 182;
    if (argc == 3) {
        width = std::atoi(argv[1]);
        height = std::atoi(argv[2]);
    } else if (argc != 1 || width <= 0 || height <= 0 ||
               width % 14 != 0 || height % 14 != 0) {
        std::cerr << "usage: dav2_backend_parity [WIDTH HEIGHT]\n";
        return 64;
    }
    std::vector<float> input(
        static_cast<std::size_t>(width) * height * 3u);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.001f);
    }
    RunResult metal;
    RunResult vulkan;
    if (!run(model_path, DAV2_CREATE_FORCE_METAL,
             input, width, height, metal) ||
        !run(model_path, DAV2_CREATE_FORCE_VULKAN,
             input, width, height, vulkan)) {
        return 1;
    }
    double absolute_error = 0.0;
    double reference_magnitude = 0.0;
    double maximum_error = 0.0;
    for (std::size_t index = 0; index < metal.depth.size(); ++index) {
        const double error = std::abs(
            static_cast<double>(metal.depth[index]) - vulkan.depth[index]);
        absolute_error += error;
        reference_magnitude += std::abs(static_cast<double>(vulkan.depth[index]));
        maximum_error = std::max(maximum_error, error);
    }
    const double relative_l1 = reference_magnitude == 0.0
        ? absolute_error : absolute_error / reference_magnitude;
    std::cout << "Metal " << metal.milliseconds << " ms, Vulkan "
              << vulkan.milliseconds << " ms, relative L1 "
              << relative_l1 * 100.0 << "%, max error "
              << maximum_error << '\n';
    return relative_l1 < 0.01 ? 0 : 2;
}
