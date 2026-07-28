#pragma once

#include "encoder.h"
#include "gpu_model.h"
#include "operators.h"
#include "vulkan.h"

#include <cstdint>
#include <string>

namespace dav2 {

struct FeatureMap {
    VulkanBuffer buffer;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
};

class DptHead {
public:
    DptHead(
        dav2_encoder encoder,
        VulkanContext& context,
        const GpuModel& weights,
        VulkanOperators& operators);

    FeatureMap forward(EncoderOutput&& encoded);

private:
    void select_convolution_block();

    FeatureMap conv(
        FeatureMap&& input,
        const std::string& weight,
        const std::string& bias,
        std::uint32_t output_channels,
        std::uint32_t kernel,
        std::uint32_t stride,
        std::uint32_t padding,
        bool has_bias);
    FeatureMap residual_unit(
        FeatureMap&& input,
        const std::string& prefix);
    FeatureMap fusion(
        FeatureMap&& path,
        FeatureMap&& skip,
        const std::string& prefix,
        std::uint32_t output_width,
        std::uint32_t output_height);

    VulkanContext& context_;
    const GpuModel& weights_;
    VulkanOperators& operators_;
    std::uint32_t embedding_ = 0;
    std::uint32_t features_ = 0;
    std::uint32_t project_channels_[4]{};
    VulkanBuffer zero_bias_;
    bool convolution_block_selected_ = false;
    bool convolution_block8_ = false;
};

}  // namespace dav2
