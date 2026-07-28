#include "operators.h"

#include "add_scaled_spv.h"
#include "add_spv.h"
#include "add_position_spv.h"
#include "bilinear_align_true_spv.h"
#include "bmm_spv.h"
#include "conv2d_spv.h"
#include "conv_transpose_nonoverlap_spv.h"
#include "gelu_spv.h"
#include "layer_norm_spv.h"
#include "linear_spv.h"
#include "merge_heads_spv.h"
#include "prepare_tokens_spv.h"
#include "position_bicubic_spv.h"
#include "project_tokens_spv.h"
#include "relu_spv.h"
#include "split_qkv_spv.h"
#include "softmax_lastdim_spv.h"

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
      gelu_(context.create_pipeline(
          dav2_gelu_spv, dav2_gelu_spv_size, 2, 4)),
      layer_norm_(context.create_pipeline(
          dav2_layer_norm_spv, dav2_layer_norm_spv_size, 4, 12)),
      add_scaled_(context.create_pipeline(
          dav2_add_scaled_spv, dav2_add_scaled_spv_size, 3, 8)),
      split_qkv_(context.create_pipeline(
          dav2_split_qkv_spv, dav2_split_qkv_spv_size, 4, 16)),
      bmm_(context.create_pipeline(
          dav2_bmm_spv, dav2_bmm_spv_size, 3, 20)),
      softmax_lastdim_(context.create_pipeline(
          dav2_softmax_lastdim_spv,
          dav2_softmax_lastdim_spv_size,
          2,
          8)),
      merge_heads_(context.create_pipeline(
          dav2_merge_heads_spv, dav2_merge_heads_spv_size, 2, 8)),
      prepare_tokens_(context.create_pipeline(
          dav2_prepare_tokens_spv,
          dav2_prepare_tokens_spv_size,
          5,
          20)),
      position_bicubic_(context.create_pipeline(
          dav2_position_bicubic_spv,
          dav2_position_bicubic_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          0)),
      add_position_(context.create_pipeline(
          dav2_add_position_spv,
          dav2_add_position_spv_size,
          4,
          16)),
      add_(context.create_pipeline(
          dav2_add_spv, dav2_add_spv_size, 3, 4)),
      project_tokens_(context.create_pipeline(
          dav2_project_tokens_spv,
          dav2_project_tokens_spv_size,
          4,
          16)),
      conv2d_(context.create_pipeline(
          dav2_conv2d_spv, dav2_conv2d_spv_size, 4, 40)),
      conv_transpose_nonoverlap_(context.create_pipeline(
          dav2_conv_transpose_nonoverlap_spv,
          dav2_conv_transpose_nonoverlap_spv_size,
          4,
          20)),
      bilinear_align_true_(context.create_pipeline(
          dav2_bilinear_align_true_spv,
          dav2_bilinear_align_true_spv_size,
          2,
          20)),
      relu_(context.create_pipeline(
          dav2_relu_spv, dav2_relu_spv_size, 2, 4)) {}

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
        linear_,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(divide_up(output_columns, 4), 8),
        divide_up(divide_up(rows, 4), 8));
    if (gelu) {
        struct GeluParameters {
            std::uint32_t count;
        } gelu_parameters{rows * output_columns};
        context_.dispatch(
            gelu_,
            {&output, &output},
            &gelu_parameters,
            sizeof(gelu_parameters),
            divide_up(gelu_parameters.count, 256));
    }
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
        {&output, &addend, &scale},
        &parameters,
        sizeof(parameters),
        divide_up(count, 256));
    struct AddParameters {
        std::uint32_t count;
    } add_parameters{count};
    context_.dispatch(
        add_,
        {&output, &residual, &output},
        &add_parameters,
        sizeof(add_parameters),
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
    const std::uint64_t score_elements =
        std::uint64_t(heads) * tokens * tokens;
    VulkanBuffer scores =
        context_.create_device_buffer(score_elements * sizeof(float));
    VulkanBuffer probabilities =
        context_.create_device_buffer(score_elements * sizeof(float));
    struct BmmParameters {
        std::uint32_t rows;
        std::uint32_t columns;
        std::uint32_t inner;
        std::uint32_t batches;
        std::uint32_t weight_transposed;
    } score_parameters{tokens, tokens, 64, heads, 1};
    context_.dispatch(
        bmm_,
        {&scores, &query, &key},
        &score_parameters,
        sizeof(score_parameters),
        divide_up(divide_up(tokens, 4), 8),
        divide_up(divide_up(tokens, 8), 8),
        heads);
    struct SoftmaxParameters {
        std::uint32_t rows;
        std::uint32_t columns;
    } softmax_parameters{heads * tokens, tokens};
    context_.dispatch(
        softmax_lastdim_,
        {&probabilities, &scores},
        &softmax_parameters,
        sizeof(softmax_parameters),
        softmax_parameters.rows);
    BmmParameters value_parameters{tokens, 64, tokens, heads, 0};
    context_.dispatch(
        bmm_,
        {&output, &probabilities, &value},
        &value_parameters,
        sizeof(value_parameters),
        divide_up(divide_up(64, 4), 8),
        divide_up(divide_up(tokens, 8), 8),
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

void VulkanOperators::prepare_tokens(
    VulkanBuffer& output,
    const VulkanBuffer& image,
    const VulkanBuffer& patch_weight,
    const VulkanBuffer& patch_bias,
    const VulkanBuffer& class_token,
    const VulkanBuffer& position,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t embedding) {
    if (input_width == 0 || input_height == 0 ||
        input_width % 14 != 0 || input_height % 14 != 0 ||
        embedding == 0) {
        throw std::invalid_argument("invalid patch embedding dimensions");
    }
    const std::uint32_t patch_width = input_width / 14;
    const std::uint32_t patch_height = input_height / 14;
    const std::uint64_t tokens =
        std::uint64_t(patch_width) * patch_height + 1;
    require_bytes(
        image, std::uint64_t(input_width) * input_height * 3, "image");
    require_bytes(
        patch_weight, std::uint64_t(embedding) * 3 * 14 * 14,
        "patch weight");
    require_bytes(patch_bias, embedding, "patch bias");
    require_bytes(class_token, embedding, "class token");
    require_bytes(
        position, std::uint64_t(1370) * embedding, "position");
    require_bytes(output, tokens * embedding, "token output");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t patch_width;
        std::uint32_t patch_height;
        std::uint32_t embedding;
    } parameters{
        input_width,
        input_height,
        patch_width,
        patch_height,
        embedding,
    };
    VulkanBuffer interpolated =
        context_.create_device_buffer(tokens * embedding * sizeof(float));
    context_.dispatch(
        prepare_tokens_,
        {
            &output,
            &image,
            &patch_weight,
            &patch_bias,
            &class_token,
        },
        &parameters,
        sizeof(parameters),
        divide_up(embedding, 8),
        divide_up(static_cast<std::uint32_t>(tokens), 8));
    struct BufferMetadata {
        std::uint32_t logical_sizes[4];
        std::uint32_t logical_strides[4];
        std::uint32_t physical_strides[4];
        std::uint32_t info[4];
    };
    const std::uint32_t spatial = patch_width * patch_height;
    const BufferMetadata output_metadata{
        {patch_width, patch_height, embedding, 1},
        {1, patch_width, spatial, spatial * embedding},
        {1, patch_width, spatial, spatial * embedding},
        {4, spatial * embedding, spatial * embedding, 0},
    };
    const BufferMetadata input_metadata{
        {37, 37, embedding, 1},
        {1, 37, 37 * 37, 37 * 37 * embedding},
        {embedding, 37 * embedding, 1, 1370 * embedding},
        {4, 1369 * embedding, 1370 * embedding, embedding},
    };
    struct PositionBlock {
        std::int32_t info[4];
        float scale[4];
    };
    const PositionBlock position_block{
        {36, 36,
         static_cast<std::int32_t>(patch_width),
         static_cast<std::int32_t>(patch_height)},
        {
            static_cast<float>(
                1.0 /
                ((static_cast<double>(patch_width) + 0.1) / 37.0)),
            static_cast<float>(
                1.0 /
                ((static_cast<double>(patch_height) + 0.1) / 37.0)),
            0.0f,
            0.0f,
        },
    };
    VulkanBuffer output_metadata_buffer =
        context_.create_host_buffer(sizeof(output_metadata));
    VulkanBuffer input_metadata_buffer =
        context_.create_host_buffer(sizeof(input_metadata));
    VulkanBuffer position_block_buffer =
        context_.create_host_buffer(sizeof(position_block));
    context_.upload(
        output_metadata_buffer, &output_metadata, sizeof(output_metadata));
    context_.upload(
        input_metadata_buffer, &input_metadata, sizeof(input_metadata));
    context_.upload(
        position_block_buffer, &position_block, sizeof(position_block));
    context_.dispatch(
        position_bicubic_,
        {
            &interpolated,
            &output_metadata_buffer,
            &position,
            &input_metadata_buffer,
            &position_block_buffer,
        },
        nullptr,
        0,
        divide_up(spatial * embedding, 256));
    struct AddPositionParameters {
        std::uint32_t patch_width;
        std::uint32_t patch_height;
        std::uint32_t embedding;
        std::uint32_t count;
    } add_parameters{
        patch_width,
        patch_height,
        embedding,
        static_cast<std::uint32_t>(tokens * embedding),
    };
    context_.dispatch(
        add_position_,
        {&output, &output, &interpolated, &position},
        &add_parameters,
        sizeof(add_parameters),
        divide_up(add_parameters.count, 256));
}

void VulkanOperators::project_tokens(
    VulkanBuffer& output,
    const VulkanBuffer& tokens,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t embedding,
    std::uint32_t output_channels) {
    if (width == 0 || height == 0 || embedding == 0 ||
        output_channels == 0) {
        throw std::invalid_argument("invalid token projection dimensions");
    }
    require_bytes(
        tokens,
        (std::uint64_t(width) * height + 1) * embedding,
        "tokens");
    require_bytes(
        weight, std::uint64_t(output_channels) * embedding, "weight");
    require_bytes(bias, output_channels, "bias");
    require_bytes(
        output,
        std::uint64_t(width) * height * output_channels,
        "output");
    struct Parameters {
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t embedding;
        std::uint32_t output_channels;
    } parameters{width, height, embedding, output_channels};
    context_.dispatch(
        project_tokens_,
        {&output, &tokens, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(width, 8),
        divide_up(height, 8),
        output_channels);
}

void VulkanOperators::conv2d(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t input_channels,
    std::uint32_t output_channels,
    std::uint32_t kernel,
    std::uint32_t stride,
    std::uint32_t padding,
    bool has_bias) {
    if (input_width == 0 || input_height == 0 || input_channels == 0 ||
        output_channels == 0 || kernel == 0 || stride == 0 ||
        input_width + 2 * padding < kernel ||
        input_height + 2 * padding < kernel) {
        throw std::invalid_argument("invalid convolution dimensions");
    }
    const std::uint32_t output_width =
        (input_width + 2 * padding - kernel) / stride + 1;
    const std::uint32_t output_height =
        (input_height + 2 * padding - kernel) / stride + 1;
    require_bytes(
        input,
        std::uint64_t(input_width) * input_height * input_channels,
        "convolution input");
    require_bytes(
        weight,
        std::uint64_t(output_channels) * input_channels * kernel * kernel,
        "convolution weight");
    require_bytes(bias, has_bias ? output_channels : 1, "convolution bias");
    require_bytes(
        output,
        std::uint64_t(output_width) * output_height * output_channels,
        "convolution output");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_width;
        std::uint32_t output_height;
        std::uint32_t output_channels;
        std::uint32_t kernel;
        std::uint32_t stride;
        std::int32_t padding;
        std::uint32_t has_bias;
    } parameters{
        input_width, input_height, input_channels,
        output_width, output_height, output_channels,
        kernel, stride, static_cast<std::int32_t>(padding),
        has_bias ? 1u : 0u,
    };
    context_.dispatch(
        conv2d_,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8),
        output_channels);
}

void VulkanOperators::conv_transpose_nonoverlap(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t input_channels,
    std::uint32_t output_channels,
    std::uint32_t kernel) {
    if (input_width == 0 || input_height == 0 || input_channels == 0 ||
        output_channels == 0 || kernel == 0) {
        throw std::invalid_argument("invalid transposed convolution dimensions");
    }
    const std::uint32_t output_width = input_width * kernel;
    const std::uint32_t output_height = input_height * kernel;
    require_bytes(
        input,
        std::uint64_t(input_width) * input_height * input_channels,
        "transposed convolution input");
    require_bytes(
        weight,
        std::uint64_t(input_channels) * output_channels * kernel * kernel,
        "transposed convolution weight");
    require_bytes(bias, output_channels, "transposed convolution bias");
    require_bytes(
        output,
        std::uint64_t(output_width) * output_height * output_channels,
        "transposed convolution output");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_channels;
        std::uint32_t kernel;
    } parameters{
        input_width, input_height, input_channels, output_channels, kernel};
    context_.dispatch(
        conv_transpose_nonoverlap_,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8),
        output_channels);
}

void VulkanOperators::bilinear_align_true(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t output_width,
    std::uint32_t output_height,
    std::uint32_t channels) {
    if (input_width == 0 || input_height == 0 || output_width == 0 ||
        output_height == 0 || channels == 0) {
        throw std::invalid_argument("invalid bilinear dimensions");
    }
    require_bytes(
        input, std::uint64_t(input_width) * input_height * channels,
        "bilinear input");
    require_bytes(
        output, std::uint64_t(output_width) * output_height * channels,
        "bilinear output");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t output_width;
        std::uint32_t output_height;
        std::uint32_t channels;
    } parameters{
        input_width, input_height, output_width, output_height, channels};
    context_.dispatch(
        bilinear_align_true_,
        {&output, &input},
        &parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8),
        channels);
}

void VulkanOperators::relu(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    std::uint32_t count) {
    require_bytes(output, count, "ReLU output");
    require_bytes(input, count, "ReLU input");
    context_.dispatch(
        relu_, {&output, &input}, &count, sizeof(count),
        divide_up(count, 256));
}

void VulkanOperators::add(
    VulkanBuffer& output,
    const VulkanBuffer& left,
    const VulkanBuffer& right,
    std::uint32_t count) {
    require_bytes(output, count, "add output");
    require_bytes(left, count, "add left");
    require_bytes(right, count, "add right");
    context_.dispatch(
        add_, {&output, &left, &right}, &count, sizeof(count),
        divide_up(count, 256));
}

}  // namespace dav2
