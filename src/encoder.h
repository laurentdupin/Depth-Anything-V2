#pragma once

#include "depth_anything_v2.h"
#include "gpu_model.h"
#include "operators.h"
#include "vulkan.h"

#include <cstdint>
#include <string>
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
        GpuModel& weights,
        VulkanOperators& operators,
        inferbridge::native::Precision precision,
        bool force_fp32_attention);

    void prepare(std::uint32_t width, std::uint32_t height);

    EncoderOutput forward(
        const VulkanBuffer& image,
        std::uint32_t width,
        std::uint32_t height);

private:
    void select_linear_tile(std::uint32_t rows);
    const VulkanBuffer& linear_weight(const std::string& name) const;
    void linear(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const std::string& weight_name,
        const std::string& bias_name,
        std::uint32_t rows,
        std::uint32_t input_columns,
        std::uint32_t output_columns,
        bool gelu);

    dav2_encoder encoder_;
    VulkanContext& context_;
    GpuModel& weights_;
    VulkanOperators& operators_;
    std::uint32_t embedding_ = 0;
    std::uint32_t heads_ = 0;
    std::uint32_t blocks_ = 0;
    std::uint32_t capture_[4]{};
    bool linear_tile_selected_ = false;
    bool linear_block16_ = false;
    bool linear_vectorized_ = false;
    std::uint32_t linear_vector_tile_ = 0;
    bool linear_half_weight_ = false;
    inferbridge::native::Precision precision_ =
        inferbridge::native::Precision::fp32;
    bool force_fp32_attention_ = false;
};

}  // namespace dav2
