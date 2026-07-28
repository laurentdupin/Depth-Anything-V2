#pragma once

#include "vulkan.h"

#include <cstdint>

namespace dav2 {

class VulkanOperators {
public:
    explicit VulkanOperators(VulkanContext& context);

    void linear(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t rows,
        std::uint32_t input_columns,
        std::uint32_t output_columns,
        bool gelu);

    void layer_norm(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t rows,
        std::uint32_t columns,
        float epsilon);

    void add_scaled(
        VulkanBuffer& output,
        const VulkanBuffer& residual,
        const VulkanBuffer& addend,
        const VulkanBuffer& scale,
        std::uint32_t count,
        std::uint32_t columns);

private:
    VulkanContext& context_;
    VulkanPipeline linear_;
    VulkanPipeline linear_gelu_;
    VulkanPipeline layer_norm_;
    VulkanPipeline add_scaled_;
};

}  // namespace dav2
