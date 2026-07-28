#pragma once

#include "depth_anything_v2.h"
#include "gpu_model.h"
#include "operators.h"
#include "vulkan.h"

#include <cstdint>
#include <vector>

namespace dav2 {

struct EncoderOutput {
    std::vector<VulkanBuffer> features;
    std::uint32_t patch_width = 0;
    std::uint32_t patch_height = 0;
    std::uint32_t tokens = 0;
    std::uint32_t embedding = 0;
};

class DinoEncoder {
public:
    DinoEncoder(
        dav2_encoder encoder,
        VulkanContext& context,
        const GpuModel& weights,
        VulkanOperators& operators);

    EncoderOutput forward(
        const VulkanBuffer& image,
        std::uint32_t width,
        std::uint32_t height);

private:
    void select_linear_tile();

    dav2_encoder encoder_;
    VulkanContext& context_;
    const GpuModel& weights_;
    VulkanOperators& operators_;
    std::uint32_t embedding_ = 0;
    std::uint32_t heads_ = 0;
    std::uint32_t blocks_ = 0;
    std::uint32_t capture_[4]{};
    bool linear_tile_selected_ = false;
    bool linear_block16_ = false;
};

}  // namespace dav2
