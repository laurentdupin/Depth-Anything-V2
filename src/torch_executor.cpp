#include "executor.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <ATen/ops/cat.h>
#include <ATen/ops/upsample_bicubic2d.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace dav2 {
namespace {

const char* encoder_name(dav2_encoder encoder) {
    switch (encoder) {
        case DAV2_ENCODER_VITS: return "vits";
        case DAV2_ENCODER_VITB: return "vitb";
        case DAV2_ENCODER_VITL: return "vitl";
        default: return "";
    }
}

void move_module_to_vulkan(
    torch::jit::Module module,
    const c10::Device& device) {
    for (const auto& parameter : module.named_parameters(/*recurse=*/false)) {
        module.setattr(
            parameter.name,
            parameter.value.to(device, /*non_blocking=*/false));
    }
    for (const auto& buffer : module.named_buffers(/*recurse=*/false)) {
        module.setattr(
            buffer.name,
            buffer.value.to(device, /*non_blocking=*/false));
    }
    for (torch::jit::Module child : module.children()) {
        move_module_to_vulkan(std::move(child), device);
    }
}

class TorchExecutor final : public Executor {
public:
    TorchExecutor(
        const std::string& model_path,
        dav2_encoder encoder,
        int vulkan_device_index)
        : device_(c10::DeviceType::Vulkan, vulkan_device_index) {
        if (!std::filesystem::is_regular_file(
                std::filesystem::u8path(model_path))) {
            throw std::runtime_error("model file does not exist");
        }

        torch::jit::ExtraFilesMap extra_files{
            {"dav2_manifest.json", std::string()}
        };
        module_ = torch::jit::load(
            model_path, c10::Device(c10::kCPU), extra_files);
        const auto manifest = extra_files.find("dav2_manifest.json");
        if (manifest == extra_files.end() || manifest->second.empty()) {
            throw std::runtime_error("model has no Depth Anything V2 manifest");
        }
        const std::string expected =
            std::string("\"encoder\":\"") + encoder_name(encoder) + "\"";
        if (manifest->second.find(expected) == std::string::npos) {
            throw std::runtime_error("model encoder does not match create options");
        }

        module_.eval();
        // torch::jit::Module::to uses Variable::set_data, which intentionally
        // rejects replacing a regular TensorImpl with Vulkan's opaque
        // TensorImpl. Python Module._apply replaces each attribute instead.
        // Reproduce that legal route without involving Python.
        c10::InferenceMode inference_mode;
        move_module_to_vulkan(module_, device_);
        base_position_ = module_.attr("base_pos_embed").toTensor();
    }

    void infer(
        const float* normalized_rgb_chw,
        int width,
        int height,
        float* depth) override {
        c10::InferenceMode inference_mode;
        const auto cpu_options =
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor input_cpu = torch::from_blob(
            const_cast<float*>(normalized_rgb_chw),
            {1, 3, height, width},
            cpu_options);
        torch::Tensor input = input_cpu.to(device_, /*non_blocking=*/false);
        torch::Tensor position = position_encoding(width, height);
        torch::Tensor output =
            module_.forward({std::move(input), std::move(position)})
                .toTensor()
                .to(torch::kCPU)
                .contiguous();

        if (output.scalar_type() != torch::kFloat32 ||
            output.dim() != 3 ||
            output.size(0) != 1 ||
            output.size(1) != height ||
            output.size(2) != width) {
            throw std::runtime_error("model returned an invalid output tensor");
        }
        const std::size_t count =
            static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        std::memcpy(depth, output.const_data_ptr<float>(), count * sizeof(float));
    }

private:
    torch::Tensor position_encoding(int width, int height) {
        const int64_t patch_height = height / 14;
        const int64_t patch_width = width / 14;
        const int64_t patch_count = base_position_.size(1) - 1;
        const int64_t side =
            static_cast<int64_t>(std::sqrt(static_cast<double>(patch_count)));
        if (side * side != patch_count) {
            throw std::runtime_error("model has invalid base position embedding");
        }

        torch::Tensor class_position = base_position_.slice(1, 0, 1);
        torch::Tensor patch_position =
            base_position_.slice(1, 1)
                .reshape({1, side, side, base_position_.size(2)})
                .permute({0, 3, 1, 2});
        const double scale_height =
            (static_cast<double>(patch_height) + 0.1) /
            static_cast<double>(side);
        const double scale_width =
            (static_cast<double>(patch_width) + 0.1) /
            static_cast<double>(side);
        patch_position = at::upsample_bicubic2d(
            patch_position,
            {patch_height, patch_width},
            /*align_corners=*/false,
            scale_height,
            scale_width);
        patch_position = patch_position.permute({0, 2, 3, 1}).reshape(
            {1, patch_height * patch_width, base_position_.size(2)});
        return at::cat({class_position, patch_position}, 1);
    }

    c10::Device device_;
    torch::jit::Module module_;
    torch::Tensor base_position_;
};

}  // namespace

std::unique_ptr<Executor> create_executor(
    const std::string& model_path,
    dav2_encoder encoder,
    int vulkan_device_index) {
    if (vulkan_device_index < 0) {
        throw std::invalid_argument("vulkan_device_index must be non-negative");
    }
    return std::make_unique<TorchExecutor>(
        model_path, encoder, vulkan_device_index);
}

}  // namespace dav2
