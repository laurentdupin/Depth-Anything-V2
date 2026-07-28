#include "operators.h"

#include "add_scaled_spv.h"
#include "layer_norm_spv.h"
#include "linear_gelu_spv.h"
#include "linear_spv.h"
#include "merge_heads_spv.h"
#include "split_qkv_spv.h"
#include "attention_head64_spv.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace dav2 {
namespace {

std::uint32_t divide_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

void require_bytes(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
        buffer.size() < elements * sizeof(float)) {
        throw std::invalid_argument(
            std::string(name) + " Vulkan buffer is too small");
    }
}

}  // namespace

VulkanOperators::VulkanOperators(VulkanContext& context)
    : context_(context),
      linear_(context.create_pipeline(
          dav2_linear_spv, dav2_linear_spv_size, 4, 12)),
      linear_gelu_(context.create_pipeline(
          dav2_linear_gelu_spv, dav2_linear_gelu_spv_size, 4, 12)),
      layer_norm_(context.create_pipeline(
          dav2_layer_norm_spv, dav2_layer_norm_spv_size, 4, 12)),
      add_scaled_(context.create_pipeline(
          dav2_add_scaled_spv, dav2_add_scaled_spv_size, 4, 8)),
      split_qkv_(context.create_pipeline(
          dav2_split_qkv_spv, dav2_split_qkv_spv_size, 4, 16)),
      attention_head64_(context.create_pipeline(
          dav2_attention_head64_spv,
          dav2_attention_head64_spv_size,
          4,
          8)),
      merge_heads_(context.create_pipeline(
          dav2_merge_heads_spv, dav2_merge_heads_spv_size, 2, 8)) {}

void VulkanOperators::linear(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool gelu) {
    if (rows == 0 || input_columns == 0 || output_columns == 0) {
        throw std::invalid_argument("linear dimensions cannot be zero");
    }
    require_bytes(input, std::uint64_t(rows) * input_columns, "input");
    require_bytes(
        weight,
        std::uint64_t(output_columns) * input_columns,
        "weight");
    require_bytes(bias, output_columns, "bias");
    require_bytes(
        output, std::uint64_t(rows) * output_columns, "output");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
    } parameters{rows, input_columns, output_columns};
    context_.dispatch(
        gelu ? linear_gelu_ : linear_,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(output_columns, 16),
        divide_up(rows, 16));
}

void VulkanOperators::layer_norm(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t columns,
    float epsilon) {
    if (rows == 0 || columns == 0 || epsilon <= 0.0f) {
        throw std::invalid_argument("invalid layer norm parameters");
    }
    require_bytes(input, std::uint64_t(rows) * columns, "input");
    require_bytes(output, std::uint64_t(rows) * columns, "output");
    require_bytes(weight, columns, "weight");
    require_bytes(bias, columns, "bias");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t columns;
        float epsilon;
    } parameters{rows, columns, epsilon};
    context_.dispatch(
        layer_norm_,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        rows);
}

void VulkanOperators::add_scaled(
    VulkanBuffer& output,
    const VulkanBuffer& residual,
    const VulkanBuffer& addend,
    const VulkanBuffer& scale,
    std::uint32_t count,
    std::uint32_t columns) {
    if (count == 0 || columns == 0 || count % columns != 0) {
        throw std::invalid_argument("invalid add-scaled dimensions");
    }
    require_bytes(output, count, "output");
    require_bytes(residual, count, "residual");
    require_bytes(addend, count, "addend");
    require_bytes(scale, columns, "scale");
    struct Parameters {
        std::uint32_t count;
        std::uint32_t columns;
    } parameters{count, columns};
    context_.dispatch(
        add_scaled_,
        {&output, &residual, &addend, &scale},
        &parameters,
        sizeof(parameters),
        divide_up(count, 256));
}

void VulkanOperators::split_qkv(
    VulkanBuffer& query,
    VulkanBuffer& key,
    VulkanBuffer& value,
    const VulkanBuffer& qkv,
    std::uint32_t tokens,
    std::uint32_t heads) {
    if (tokens == 0 || heads == 0) {
        throw std::invalid_argument("invalid QKV dimensions");
    }
    const std::uint64_t embedding = std::uint64_t(heads) * 64;
    const std::uint64_t elements = std::uint64_t(tokens) * embedding;
    require_bytes(qkv, elements * 3, "QKV");
    require_bytes(query, elements, "query");
    require_bytes(key, elements, "key");
    require_bytes(value, elements, "value");
    struct Parameters {
        std::uint32_t tokens;
        std::uint32_t heads;
        std::uint32_t embedding;
        float query_scale;
    } parameters{tokens, heads, heads * 64, 0.125f};
    context_.dispatch(
        split_qkv_,
        {&query, &key, &value, &qkv},
        &parameters,
        sizeof(parameters),
        8,
        divide_up(tokens, 8),
        heads);
}

void VulkanOperators::attention_head64(
    VulkanBuffer& output,
    const VulkanBuffer& query,
    const VulkanBuffer& key,
    const VulkanBuffer& value,
    std::uint32_t tokens,
    std::uint32_t heads) {
    if (tokens == 0 || heads == 0) {
        throw std::invalid_argument("invalid attention dimensions");
    }
    const std::uint64_t elements =
        std::uint64_t(tokens) * heads * 64;
    require_bytes(output, elements, "attention output");
    require_bytes(query, elements, "query");
    require_bytes(key, elements, "key");
    require_bytes(value, elements, "value");
    struct Parameters {
        std::uint32_t tokens;
        std::uint32_t heads;
    } parameters{tokens, heads};
    context_.dispatch(
        attention_head64_,
        {&output, &query, &key, &value},
        &parameters,
        sizeof(parameters),
        1,
        divide_up(tokens, 4),
        heads);
}

void VulkanOperators::merge_heads(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    std::uint32_t tokens,
    std::uint32_t heads) {
    if (tokens == 0 || heads == 0) {
        throw std::invalid_argument("invalid head merge dimensions");
    }
    const std::uint64_t elements =
        std::uint64_t(tokens) * heads * 64;
    require_bytes(output, elements, "merged output");
    require_bytes(input, elements, "attention input");
    struct Parameters {
        std::uint32_t tokens;
        std::uint32_t heads;
    } parameters{tokens, heads};
    context_.dispatch(
        merge_heads_,
        {&output, &input},
        &parameters,
        sizeof(parameters),
        8,
        divide_up(tokens, 8),
        heads);
}

}  // namespace dav2
