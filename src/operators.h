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

    void split_qkv(
        VulkanBuffer& query,
        VulkanBuffer& key,
        VulkanBuffer& value,
        const VulkanBuffer& qkv,
        std::uint32_t tokens,
        std::uint32_t heads);

    void attention_head64(
        VulkanBuffer& output,
        const VulkanBuffer& query,
        const VulkanBuffer& key,
        const VulkanBuffer& value,
        std::uint32_t tokens,
        std::uint32_t heads);

    void merge_heads(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        std::uint32_t tokens,
        std::uint32_t heads);

    void prepare_tokens(
        VulkanBuffer& output,
        const VulkanBuffer& image,
        const VulkanBuffer& patch_weight,
        const VulkanBuffer& patch_bias,
        const VulkanBuffer& class_token,
        const VulkanBuffer& position,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t embedding);

private:
    VulkanContext& context_;
    VulkanPipeline linear_;
    VulkanPipeline gelu_;
    VulkanPipeline layer_norm_;
    VulkanPipeline add_scaled_;
    VulkanPipeline split_qkv_;
    VulkanPipeline bmm_;
    VulkanPipeline softmax_lastdim_;
    VulkanPipeline merge_heads_;
    VulkanPipeline prepare_tokens_;
    VulkanPipeline position_bicubic_;
    VulkanPipeline add_;
};

}  // namespace dav2
