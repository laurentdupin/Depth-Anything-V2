#include "depth_anything_v2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct RunResult {
    std::vector<float> depth;
    double milliseconds = 0.0;
};

bool run(
    const char* model_path, std::uint32_t backend_flag,
    std::uint32_t precision_flag,
    const std::vector<float>& input, int width, int height,
    RunResult& result) {
    const dav2_create_options options{
        sizeof(options), DAV2_ABI_VERSION, DAV2_ENCODER_VITS, 0,
        backend_flag | precision_flag};
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
    std::vector<double> samples;
    for (int iteration = 0; status == DAV2_STATUS_OK && iteration < 5;
         ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        status = dav2_infer_tensor_f32(
            context, input.data(), width, height,
            result.depth.data(), result.depth.size());
        samples.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    if (status != DAV2_STATUS_OK) {
        std::cerr << "infer failed: " << dav2_last_error() << '\n';
        dav2_destroy(context);
        return false;
    }
    std::sort(samples.begin(), samples.end());
    result.milliseconds = samples[samples.size() / 2u];
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
    if (argc == 3 || argc == 4) {
        width = std::atoi(argv[1]);
        height = std::atoi(argv[2]);
    } else if (argc != 1) {
        std::cerr << "usage: dav2_backend_parity [WIDTH HEIGHT [fp16]]\n";
        return 64;
    }
    if (width <= 0 || height <= 0 ||
        width % 14 != 0 || height % 14 != 0) {
        std::cerr << "usage: dav2_backend_parity [WIDTH HEIGHT [fp16]]\n";
        return 64;
    }
    const bool fp16 = argc == 4 && std::string(argv[3]) == "fp16";
    if (argc == 4 && !fp16) {
        std::cerr << "precision must be fp16\n";
        return 64;
    }
    const std::uint32_t precision_flag = fp16
        ? DAV2_CREATE_FORCE_FP16 : DAV2_CREATE_FORCE_FP32;
    std::vector<float> input(
        static_cast<std::size_t>(width) * height * 3u);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.001f);
    }
    RunResult metal;
    RunResult vulkan;
    if (!run(model_path, DAV2_CREATE_FORCE_METAL, precision_flag,
             input, width, height, metal) ||
        !run(model_path, DAV2_CREATE_FORCE_VULKAN, precision_flag,
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
    std::cout << (fp16 ? "FP16 " : "FP32 ")
              << "Metal " << metal.milliseconds << " ms, Vulkan "
              << vulkan.milliseconds << " ms, relative L1 "
              << relative_l1 * 100.0 << "%, max error "
              << maximum_error << '\n';
    return relative_l1 < 0.01 ? 0 : 2;
}
