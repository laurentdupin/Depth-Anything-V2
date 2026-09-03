#include "operators.h"

#include "add_scaled_spv.h"
#include "add_spv.h"
#include "add_position_spv.h"
#include "bilinear_align_true_spv.h"
#include "bilinear_align_true_image_spv.h"
#include "normalize_relative_spv.h"
#include "normalize_metric_spv.h"
#include "reduce_minmax_spv.h"
#include "bmm_spv.h"
#include "bmm_score_half_spv.h"
#include "bmm_value_half_spv.h"
#include "bmm_score_fp16_spv.h"
#include "bmm_value_fp16_spv.h"
#include "bmm_score_coop_fp16_spv.h"
#include "conv2d_spv.h"
#include "conv2d_pointwise_gemm_spv.h"
#include "conv2d_pointwise_gemm_half_spv.h"
#include "conv2d_pointwise_coop_workgroup_nv_fp16_spv.h"
#include "conv2d_tiled16x8_spv.h"
#include "conv2d_tiled16x8_half_spv.h"
#include "conv2d_stride2_tiled8x8_spv.h"
#include "conv2d_stride2_tiled8x8_half_spv.h"
#include "conv2d8_spv.h"
#include "conv2d_half_spv.h"
#include "conv2d_fp16_spv.h"
#include "conv2d8_half_spv.h"
#include "conv2d_tiled_spv.h"
#include "conv2d8_tiled_spv.h"
#include "conv2d_tiled_half_spv.h"
#include "conv2d8_tiled_half_spv.h"
#include "conv_transpose_nonoverlap_spv.h"
#include "conv_transpose_nonoverlap_half_spv.h"
#include "conv_transpose_nonoverlap_fp16_spv.h"
#include "im2col_quantize_int8_spv.h"
#include "conv_transpose_int8_spv.h"
#include "gelu_spv.h"
#include "layer_norm_spv.h"
#include "layer_norm_fp16_spv.h"
#include "linear_spv.h"
#include "linear16_spv.h"
#include "linear_half_spv.h"
#include "linear16_half_spv.h"
#include "linear_vec4_spv.h"
#include "linear_vec4_half_spv.h"
#include "linear_vec8_spv.h"
#include "linear_vec8_half_spv.h"
#include "linear_vec8_residual_spv.h"
#include "linear_vec8_half_residual_spv.h"
#include "linear_vec16_spv.h"
#include "linear_vec16_half_spv.h"
#include "linear_vec16_gelu_spv.h"
#include "linear_vec16_residual_spv.h"
#include "linear_vec16_half_residual_spv.h"
#include "pack_fp16_spv.h"
#include "linear_vec16_fp16_spv.h"
#include "linear_vec16_fp16_output_gelu_spv.h"
#include "linear_coop_fp16_spv.h"
#include "linear_coop_tall_fp16_spv.h"
#include "linear_coop_fp16_output_gelu_spv.h"
#include "linear_coop_workgroup_nv_fp16_spv.h"
#include "linear_coop_workgroup_nv_fp16_output_gelu_spv.h"
#include "linear_coop_workgroup_nv_fp16_transposed_spv.h"
#include "linear_coop_workgroup_nv_fp16_transposed_output_gelu_spv.h"
#include "quantize_rows_int8_fused_spv.h"
#include "linear_int8_tiled_spv.h"
#include "linear_coop_workgroup_nv_int8_transposed_spv.h"
#include "prepare_tokens_spv.h"
#include "prepare_tokens_fp16_spv.h"
#include "copy_class_token_spv.h"
#include "position_bicubic_spv.h"
#include "project_tokens_spv.h"
#include "project_tokens_half_spv.h"
#include "project_tokens_fp16_spv.h"
#include "relu_spv.h"
#include "sigmoid_scale_spv.h"
#include "softmax_lastdim_spv.h"
#include "softmax_lastdim_half_spv.h"

#include <cstdlib>
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

void require_half_elements(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    const std::uint64_t words = (elements + 1) / 2;
    if (words > std::numeric_limits<std::uint64_t>::max() /
            sizeof(std::uint32_t) ||
        buffer.size() < words * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            std::string(name) + " packed-half Vulkan buffer is too small");
    }
}

}  // namespace

VulkanOperators::VulkanOperators(VulkanContext& context)
    : context_(context),
      linear_(context.create_pipeline(
          dav2_linear_spv, dav2_linear_spv_size, 4, 12)),
      linear16_(context.create_pipeline(
          dav2_linear16_spv, dav2_linear16_spv_size, 4, 12)),
      linear_half_(context.create_pipeline(
          dav2_linear_half_spv, dav2_linear_half_spv_size, 4, 12)),
      linear16_half_(context.create_pipeline(
          dav2_linear16_half_spv,
          dav2_linear16_half_spv_size,
          4,
          12)),
      linear_vec4_(context.create_pipeline(
          dav2_linear_vec4_spv,
          dav2_linear_vec4_spv_size,
          4,
          12)),
      linear_vec4_half_(context.create_pipeline(
          dav2_linear_vec4_half_spv,
          dav2_linear_vec4_half_spv_size,
          4,
          12)),
      linear_vec8_(context.create_pipeline(
          dav2_linear_vec8_spv,
          dav2_linear_vec8_spv_size,
          4,
          12)),
      linear_vec8_half_(context.create_pipeline(
          dav2_linear_vec8_half_spv,
          dav2_linear_vec8_half_spv_size,
          4,
          12)),
      linear_vec8_residual_(context.create_pipeline(
          dav2_linear_vec8_residual_spv,
          dav2_linear_vec8_residual_spv_size,
          6,
          12)),
      linear_vec8_half_residual_(context.create_pipeline(
          dav2_linear_vec8_half_residual_spv,
          dav2_linear_vec8_half_residual_spv_size,
          6,
          12)),
      linear_vec16_(context.create_pipeline(
          dav2_linear_vec16_spv,
          dav2_linear_vec16_spv_size,
          4,
          12)),
      linear_vec16_half_(context.create_pipeline(
          dav2_linear_vec16_half_spv,
          dav2_linear_vec16_half_spv_size,
          4,
          12)),
      linear_vec16_gelu_(context.create_pipeline(
          dav2_linear_vec16_gelu_spv,
          dav2_linear_vec16_gelu_spv_size,
          4,
          12)),
      linear_vec16_residual_(context.create_pipeline(
          dav2_linear_vec16_residual_spv,
          dav2_linear_vec16_residual_spv_size,
          6,
          12)),
      linear_vec16_half_residual_(context.create_pipeline(
          dav2_linear_vec16_half_residual_spv,
          dav2_linear_vec16_half_residual_spv_size,
          6,
          12)),
      gelu_(context.create_pipeline(
          dav2_gelu_spv, dav2_gelu_spv_size, 2, 4)),
      layer_norm_(context.create_pipeline(
          dav2_layer_norm_spv, dav2_layer_norm_spv_size, 4, 12)),
      add_scaled_(context.create_pipeline(
          dav2_add_scaled_spv, dav2_add_scaled_spv_size, 4, 8)),
      bmm_(context.create_pipeline(
          dav2_bmm_spv, dav2_bmm_spv_size, 3, 36)),
      bmm_score_half_(context.create_pipeline(
          dav2_bmm_score_half_spv,
          dav2_bmm_score_half_spv_size,
          2,
          8)),
      bmm_value_half_(context.create_pipeline(
          dav2_bmm_value_half_spv,
          dav2_bmm_value_half_spv_size,
          3,
          8)),
      softmax_lastdim_(context.create_pipeline(
          dav2_softmax_lastdim_spv,
          dav2_softmax_lastdim_spv_size,
          2,
          8)),
      softmax_lastdim_half_(context.create_pipeline(
          dav2_softmax_lastdim_half_spv,
          dav2_softmax_lastdim_half_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {
              VK_ACCESS_SHADER_READ_BIT |
              VK_ACCESS_SHADER_WRITE_BIT,
          },
          8)),
      prepare_tokens_(context.create_pipeline(
          dav2_prepare_tokens_spv,
          dav2_prepare_tokens_spv_size,
          5,
          20)),
      copy_class_token_(context.create_pipeline(
          dav2_copy_class_token_spv,
          dav2_copy_class_token_spv_size,
          2,
          4)),
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
      project_tokens_half_(context.create_pipeline(
          dav2_project_tokens_half_spv,
          dav2_project_tokens_half_spv_size,
          4,
          16)),
      conv2d_(context.create_pipeline(
          dav2_conv2d_spv, dav2_conv2d_spv_size, 4, 40)),
      conv2d_pointwise_gemm_(context.create_pipeline(
          dav2_conv2d_pointwise_gemm_spv,
          dav2_conv2d_pointwise_gemm_spv_size, 4, 40)),
      conv2d_pointwise_gemm_half_(context.create_pipeline(
          dav2_conv2d_pointwise_gemm_half_spv,
          dav2_conv2d_pointwise_gemm_half_spv_size, 4, 40)),
      conv2d_tiled16x8_(context.create_pipeline(
          dav2_conv2d_tiled16x8_spv,
          dav2_conv2d_tiled16x8_spv_size, 4, 40)),
      conv2d_tiled16x8_half_(context.create_pipeline(
          dav2_conv2d_tiled16x8_half_spv,
          dav2_conv2d_tiled16x8_half_spv_size, 4, 40)),
      conv2d_stride2_tiled8x8_(context.create_pipeline(
          dav2_conv2d_stride2_tiled8x8_spv,
          dav2_conv2d_stride2_tiled8x8_spv_size, 4, 40)),
      conv2d_stride2_tiled8x8_half_(context.create_pipeline(
          dav2_conv2d_stride2_tiled8x8_half_spv,
          dav2_conv2d_stride2_tiled8x8_half_spv_size, 4, 40)),
      conv2d8_(context.create_pipeline(
          dav2_conv2d8_spv, dav2_conv2d8_spv_size, 4, 40)),
      conv2d_half_(context.create_pipeline(
          dav2_conv2d_half_spv, dav2_conv2d_half_spv_size, 4, 40)),
      conv2d8_half_(context.create_pipeline(
          dav2_conv2d8_half_spv,
          dav2_conv2d8_half_spv_size,
          4,
          40)),
      conv2d_tiled_(context.create_pipeline(
          dav2_conv2d_tiled_spv,
          dav2_conv2d_tiled_spv_size,
          4,
          40)),
      conv2d8_tiled_(context.create_pipeline(
          dav2_conv2d8_tiled_spv,
          dav2_conv2d8_tiled_spv_size,
          4,
          40)),
      conv2d_tiled_half_(context.create_pipeline(
          dav2_conv2d_tiled_half_spv,
          dav2_conv2d_tiled_half_spv_size,
          4,
          40)),
      conv2d8_tiled_half_(context.create_pipeline(
          dav2_conv2d8_tiled_half_spv,
          dav2_conv2d8_tiled_half_spv_size,
          4,
          40)),
      conv_transpose_nonoverlap_(context.create_pipeline(
          dav2_conv_transpose_nonoverlap_spv,
          dav2_conv_transpose_nonoverlap_spv_size,
          4,
          20)),
      conv_transpose_nonoverlap_half_(context.create_pipeline(
          dav2_conv_transpose_nonoverlap_half_spv,
          dav2_conv_transpose_nonoverlap_half_spv_size,
          4,
          20)),
      bilinear_align_true_(context.create_pipeline(
          dav2_bilinear_align_true_spv,
          dav2_bilinear_align_true_spv_size,
          2,
          20)),
      bilinear_align_true_image_(context.create_pipeline(
          dav2_bilinear_align_true_image_spv,
          dav2_bilinear_align_true_image_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          },
          16)),
      reduce_minmax_(context.create_pipeline(
          dav2_reduce_minmax_spv, dav2_reduce_minmax_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}, 4)),
      normalize_relative_(context.create_pipeline(
          dav2_normalize_relative_spv,
          dav2_normalize_relative_spv_size, 2, 4)),
      normalize_metric_(context.create_pipeline(
          dav2_normalize_metric_spv,
          dav2_normalize_metric_spv_size, 2, 8)),
      relu_(context.create_pipeline(
          dav2_relu_spv, dav2_relu_spv_size, 2, 4)),
      sigmoid_scale_(context.create_pipeline(
          dav2_sigmoid_scale_spv,
          dav2_sigmoid_scale_spv_size, 2, 8)) {
    if (context_.compute_capabilities().fp16) {
        pack_fp16_ = context_.create_pipeline(
            dav2_pack_fp16_spv, dav2_pack_fp16_spv_size, 2, 4);
        linear_vec16_fp16_ = context_.create_pipeline(
            dav2_linear_vec16_fp16_spv,
            dav2_linear_vec16_fp16_spv_size, 4, 12);
        linear_vec16_fp16_output_gelu_ = context_.create_pipeline(
            dav2_linear_vec16_fp16_output_gelu_spv,
            dav2_linear_vec16_fp16_output_gelu_spv_size, 4, 12);
        layer_norm_fp16_ = context_.create_pipeline(
            dav2_layer_norm_fp16_spv,
            dav2_layer_norm_fp16_spv_size, 4, 12);
        pack_fp16_.set_debug_name("pack_fp16");
        linear_vec16_fp16_.set_debug_name("linear_vec16_fp16");
        linear_vec16_fp16_output_gelu_.set_debug_name(
            "linear_vec16_fp16_output_gelu");
        layer_norm_fp16_.set_debug_name("layer_norm_fp16");
        if (context_.compute_capabilities().cooperative_matrix_fp16) {
            linear_coop_fp16_output_gelu_ = context_.create_pipeline(
                dav2_linear_coop_fp16_output_gelu_spv,
                dav2_linear_coop_fp16_output_gelu_spv_size, 4, 12);
            linear_coop_fp16_output_gelu_.set_debug_name(
                "linear_coop_fp16_output_gelu");
        }
    }
    if (context_.compute_capabilities().packed_int8_dot) {
        quantize_rows_int8_fused_ = context_.create_pipeline(
            dav2_quantize_rows_int8_fused_spv,
            dav2_quantize_rows_int8_fused_spv_size, 3, 12);
        linear_int8_tiled_ = context_.create_pipeline(
            dav2_linear_int8_tiled_spv,
            dav2_linear_int8_tiled_spv_size, 6, 28);
        im2col_quantize_int8_ = context_.create_pipeline(
            dav2_im2col_quantize_int8_spv,
            dav2_im2col_quantize_int8_spv_size, 3, 32);
        conv_transpose_int8_ = context_.create_pipeline(
            dav2_conv_transpose_int8_spv,
            dav2_conv_transpose_int8_spv_size, 6, 20);
        quantize_rows_int8_fused_.set_debug_name(
            "quantize_rows_int8_fused");
        linear_int8_tiled_.set_debug_name("linear_int8_tiled");
        im2col_quantize_int8_.set_debug_name("im2col_quantize_int8");
        conv_transpose_int8_.set_debug_name("conv_transpose_int8");
    }
    if (context_.compute_capabilities().fp16) {
        prepare_tokens_fp16_ = context_.create_pipeline(
            dav2_prepare_tokens_fp16_spv,
            dav2_prepare_tokens_fp16_spv_size, 5, 20);
        bmm_score_fp16_ = context_.create_pipeline(
            dav2_bmm_score_fp16_spv,
            dav2_bmm_score_fp16_spv_size, 2, 8);
        bmm_value_fp16_ = context_.create_pipeline(
            dav2_bmm_value_fp16_spv,
            dav2_bmm_value_fp16_spv_size, 3, 8);
        project_tokens_fp16_ = context_.create_pipeline(
            dav2_project_tokens_fp16_spv,
            dav2_project_tokens_fp16_spv_size, 4, 16);
        conv2d_fp16_ = context_.create_pipeline(
            dav2_conv2d_fp16_spv,
            dav2_conv2d_fp16_spv_size, 4, 40);
        conv_transpose_nonoverlap_fp16_ = context_.create_pipeline(
            dav2_conv_transpose_nonoverlap_fp16_spv,
            dav2_conv_transpose_nonoverlap_fp16_spv_size, 4, 20);
        prepare_tokens_fp16_.set_debug_name("prepare_tokens_fp16");
        bmm_score_fp16_.set_debug_name("bmm_score_fp16");
        bmm_value_fp16_.set_debug_name("bmm_value_fp16");
        project_tokens_fp16_.set_debug_name("project_tokens_fp16");
        conv2d_fp16_.set_debug_name("conv2d_fp16");
        conv_transpose_nonoverlap_fp16_.set_debug_name(
            "conv_transpose_nonoverlap_fp16");
    }
    if (context_.compute_capabilities().cooperative_matrix_fp16) {
        bmm_score_coop_fp16_ = context_.create_pipeline(
            dav2_bmm_score_coop_fp16_spv,
            dav2_bmm_score_coop_fp16_spv_size, 2, 8);
        bmm_score_coop_fp16_.set_debug_name("bmm_score_coop_fp16");
        linear_coop_fp16_ = context_.create_pipeline(
            dav2_linear_coop_fp16_spv,
            dav2_linear_coop_fp16_spv_size, 4, 12);
        linear_coop_fp16_.set_debug_name("linear_coop_fp16");
        linear_coop_tall_fp16_ = context_.create_pipeline(
            dav2_linear_coop_tall_fp16_spv,
            dav2_linear_coop_tall_fp16_spv_size, 4, 12);
        linear_coop_tall_fp16_.set_debug_name("linear_coop_tall_fp16");
    }
    if (context_.compute_capabilities().cooperative_matrix_workgroup_nv) {
        conv2d_pointwise_coop_workgroup_nv_fp16_ = context_.create_pipeline(
            dav2_conv2d_pointwise_coop_workgroup_nv_fp16_spv,
            dav2_conv2d_pointwise_coop_workgroup_nv_fp16_spv_size, 4, 12);
        conv2d_pointwise_coop_workgroup_nv_fp16_.set_debug_name(
            "conv2d_pointwise_coop_workgroup_nv_fp16");
        linear_coop_workgroup_nv_fp16_ = context_.create_pipeline(
            dav2_linear_coop_workgroup_nv_fp16_spv,
            dav2_linear_coop_workgroup_nv_fp16_spv_size, 4, 12);
        linear_coop_workgroup_nv_fp16_.set_debug_name(
            "linear_coop_workgroup_nv_fp16");
        linear_coop_workgroup_nv_fp16_transposed_ =
            context_.create_pipeline(
                dav2_linear_coop_workgroup_nv_fp16_transposed_spv,
                dav2_linear_coop_workgroup_nv_fp16_transposed_spv_size,
                4, 12);
        linear_coop_workgroup_nv_fp16_transposed_.set_debug_name(
            "linear_coop_workgroup_nv_fp16_transposed");
    }
    if (context_.compute_capabilities().cooperative_matrix_workgroup_int8_nv) {
        linear_coop_workgroup_nv_int8_transposed_ = context_.create_pipeline(
            dav2_linear_coop_workgroup_nv_int8_transposed_spv,
            dav2_linear_coop_workgroup_nv_int8_transposed_spv_size, 6, 12);
        linear_coop_workgroup_nv_int8_transposed_.set_debug_name(
            "linear_coop_workgroup_nv_int8_transposed");
    }
    if (context_.compute_capabilities()
            .cooperative_matrix_workgroup_epilogue_nv) {
        linear_coop_workgroup_nv_fp16_output_gelu_ =
            context_.create_pipeline(
                dav2_linear_coop_workgroup_nv_fp16_output_gelu_spv,
                dav2_linear_coop_workgroup_nv_fp16_output_gelu_spv_size,
                4, 12);
        linear_coop_workgroup_nv_fp16_output_gelu_.set_debug_name(
            "linear_coop_workgroup_nv_fp16_output_gelu");
        linear_coop_workgroup_nv_fp16_transposed_output_gelu_ =
            context_.create_pipeline(
                dav2_linear_coop_workgroup_nv_fp16_transposed_output_gelu_spv,
                dav2_linear_coop_workgroup_nv_fp16_transposed_output_gelu_spv_size,
                4, 12);
        linear_coop_workgroup_nv_fp16_transposed_output_gelu_.set_debug_name(
            "linear_coop_workgroup_nv_fp16_transposed_output_gelu");
    }
    linear_.set_debug_name("linear");
    linear16_.set_debug_name("linear16");
    linear_half_.set_debug_name("linear_half");
    linear16_half_.set_debug_name("linear16_half");
    linear_vec4_.set_debug_name("linear_vec4");
    linear_vec4_half_.set_debug_name("linear_vec4_half");
    linear_vec8_.set_debug_name("linear_vec8");
    linear_vec8_half_.set_debug_name("linear_vec8_half");
    linear_vec8_residual_.set_debug_name("linear_vec8_residual");
    linear_vec8_half_residual_.set_debug_name(
        "linear_vec8_half_residual");
    linear_vec16_.set_debug_name("linear_vec16");
    linear_vec16_half_.set_debug_name("linear_vec16_half");
    linear_vec16_gelu_.set_debug_name("linear_vec16_gelu");
    linear_vec16_residual_.set_debug_name("linear_vec16_residual");
    linear_vec16_half_residual_.set_debug_name(
        "linear_vec16_half_residual");
    gelu_.set_debug_name("gelu");
    layer_norm_.set_debug_name("layer_norm");
    add_scaled_.set_debug_name("add_scaled");
    bmm_.set_debug_name("bmm");
    bmm_score_half_.set_debug_name("bmm_score_half");
    bmm_value_half_.set_debug_name("bmm_value_half");
    softmax_lastdim_.set_debug_name("softmax_lastdim");
    softmax_lastdim_half_.set_debug_name(
        "softmax_lastdim_half");
    prepare_tokens_.set_debug_name("prepare_tokens");
    copy_class_token_.set_debug_name("copy_class_token");
    position_bicubic_.set_debug_name("position_bicubic");
    add_position_.set_debug_name("add_position");
    add_.set_debug_name("add");
    project_tokens_.set_debug_name("project_tokens");
    project_tokens_half_.set_debug_name(
        "project_tokens_half");
    conv2d_.set_debug_name("conv2d");
    conv2d_pointwise_gemm_.set_debug_name(
        "conv2d_pointwise_gemm");
    conv2d_pointwise_gemm_half_.set_debug_name(
        "conv2d_pointwise_gemm_half");
    conv2d_tiled16x8_.set_debug_name("conv2d_tiled16x8");
    conv2d_tiled16x8_half_.set_debug_name("conv2d_tiled16x8_half");
    conv2d_stride2_tiled8x8_.set_debug_name(
        "conv2d_stride2_tiled8x8");
    conv2d_stride2_tiled8x8_half_.set_debug_name(
        "conv2d_stride2_tiled8x8_half");
    conv2d8_.set_debug_name("conv2d8");
    conv2d_half_.set_debug_name("conv2d_half");
    conv2d8_half_.set_debug_name("conv2d8_half");
    conv2d_tiled_.set_debug_name("conv2d_tiled");
    conv2d8_tiled_.set_debug_name("conv2d8_tiled");
    conv2d_tiled_half_.set_debug_name("conv2d_tiled_half");
    conv2d8_tiled_half_.set_debug_name("conv2d8_tiled_half");
    conv_transpose_nonoverlap_.set_debug_name(
        "conv_transpose_nonoverlap");
    conv_transpose_nonoverlap_half_.set_debug_name(
        "conv_transpose_nonoverlap_half");
    bilinear_align_true_.set_debug_name(
        "bilinear_align_true");
    bilinear_align_true_image_.set_debug_name(
        "bilinear_align_true_image");
    reduce_minmax_.set_debug_name("reduce_minmax");
    normalize_relative_.set_debug_name("normalize_relative");
    normalize_metric_.set_debug_name("normalize_metric");
    relu_.set_debug_name("relu");
    sigmoid_scale_.set_debug_name("sigmoid_scale");
}

void VulkanOperators::linear_fp16(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& half_weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool gelu,
    bool weight_transposed) {
    if (!context_.compute_capabilities().fp16) {
        throw std::runtime_error(
            "native FP16 linear operation is unavailable on this Vulkan device");
    }
    if (rows == 0 || input_columns == 0 || output_columns == 0 ||
        input_columns % 4 != 0) {
        throw std::invalid_argument("invalid FP16 linear dimensions");
    }
    const std::uint64_t input_elements =
        std::uint64_t(rows) * input_columns;
    require_bytes(input, input_elements, "input");
    require_half_elements(
        half_weight,
        std::uint64_t(output_columns) * input_columns,
        "weight");
    require_bytes(bias, output_columns, "bias");
    require_bytes(output, std::uint64_t(rows) * output_columns, "output");
    VulkanBuffer& packed_input = fp16_workspace_.ensure(
        input_elements * sizeof(std::uint16_t),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    struct Count { std::uint32_t count; } count{
        static_cast<std::uint32_t>(input_elements)};
    context_.dispatch(
        pack_fp16_, {&packed_input, &input}, &count, sizeof(count),
        divide_up(count.count, 256));
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
    } parameters{rows, input_columns, output_columns};
    const char* disable_matrix2 =
        std::getenv("DAV2_DISABLE_MATRIX2_FP32_INPUT");
    const bool workgroup_nv = !gelu && weight_transposed && rows >= 128 &&
        context_.compute_capabilities().cooperative_matrix_workgroup_nv &&
        (disable_matrix2 == nullptr || disable_matrix2[0] == '\0' ||
         disable_matrix2[0] == '0');
    context_.dispatch(
        workgroup_nv
            ? linear_coop_workgroup_nv_fp16_transposed_
            : linear_vec16_fp16_,
        {&output, &packed_input, &half_weight, &bias},
        &parameters, sizeof(parameters),
        divide_up(output_columns, workgroup_nv ? 32 : 64),
        divide_up(rows, workgroup_nv ? 32 : 64));
    if (gelu) {
        struct GeluParameters { std::uint32_t count; } gelu_parameters{
            rows * output_columns};
        context_.dispatch(
            gelu_, {&output, &output}, &gelu_parameters,
            sizeof(gelu_parameters), divide_up(gelu_parameters.count, 256));
    }
}

void VulkanOperators::linear_fp16_half_output_gelu(
    VulkanBuffer& half_output,
    const VulkanBuffer& input,
    const VulkanBuffer& half_weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool weight_transposed) {
    if (!context_.compute_capabilities().fp16) {
        throw std::runtime_error(
            "native FP16 linear operation is unavailable on this Vulkan device");
    }
    if (rows == 0 || input_columns == 0 || output_columns == 0 ||
        input_columns % 4 != 0 || output_columns % 4 != 0) {
        throw std::invalid_argument("invalid FP16 half-output linear dimensions");
    }
    const std::uint64_t input_elements =
        std::uint64_t(rows) * input_columns;
    require_bytes(input, input_elements, "input");
    require_half_elements(
        half_weight,
        std::uint64_t(output_columns) * input_columns,
        "weight");
    require_bytes(bias, output_columns, "bias");
    require_half_elements(
        half_output, std::uint64_t(rows) * output_columns, "output");
    VulkanBuffer& packed_input = fp16_workspace_.ensure(
        input_elements * sizeof(std::uint16_t),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    struct Count { std::uint32_t count; } count{
        static_cast<std::uint32_t>(input_elements)};
    context_.dispatch(
        pack_fp16_, {&packed_input, &input}, &count, sizeof(count),
        divide_up(count.count, 256));
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
    } parameters{rows, input_columns, output_columns};
    const bool cooperative =
        context_.compute_capabilities().cooperative_matrix_fp16;
    const bool workgroup_nv = rows >= 128 && context_.compute_capabilities()
        .cooperative_matrix_workgroup_epilogue_nv;
    context_.dispatch(
        workgroup_nv
            ? (weight_transposed
                ? linear_coop_workgroup_nv_fp16_transposed_output_gelu_
                : linear_coop_workgroup_nv_fp16_output_gelu_)
            : cooperative
            ? linear_coop_fp16_output_gelu_
            : linear_vec16_fp16_output_gelu_,
        {&half_output, &packed_input, &half_weight, &bias},
        &parameters, sizeof(parameters),
        divide_up(output_columns, workgroup_nv ? 32 : cooperative ? 16 : 64),
        divide_up(rows, workgroup_nv ? 32 : cooperative ? 16 : 64));
}

void VulkanOperators::linear_fp16_half_input(
    VulkanBuffer& output,
    const VulkanBuffer& half_input,
    const VulkanBuffer& half_weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool weight_transposed) {
    if (!context_.compute_capabilities().fp16) {
        throw std::runtime_error(
            "native FP16 linear operation is unavailable on this Vulkan device");
    }
    if (rows == 0 || input_columns == 0 || output_columns == 0 ||
        input_columns % 4 != 0) {
        throw std::invalid_argument("invalid FP16 half-input linear dimensions");
    }
    require_half_elements(
        half_input, std::uint64_t(rows) * input_columns, "input");
    require_half_elements(
        half_weight,
        std::uint64_t(output_columns) * input_columns,
        "weight");
    require_bytes(bias, output_columns, "bias");
    require_bytes(output, std::uint64_t(rows) * output_columns, "output");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
    } parameters{rows, input_columns, output_columns};
    const bool cooperative =
        context_.compute_capabilities().cooperative_matrix_fp16;
    const bool workgroup_nv =
        rows >= 128 &&
        context_.compute_capabilities().cooperative_matrix_workgroup_nv;
    // Sharing each weight tile across four subgroups wins once there are
    // enough rows to amortize the larger workgroup. Small token grids retain
    // the higher-occupancy one-subgroup cooperative tile.
    const bool tall = cooperative && rows >= 256;
    if (workgroup_nv) {
        context_.dispatch(
            weight_transposed
                ? linear_coop_workgroup_nv_fp16_transposed_
                : linear_coop_workgroup_nv_fp16_,
            {&output, &half_input, &half_weight, &bias},
            &parameters, sizeof(parameters),
            divide_up(output_columns, 32), divide_up(rows, 32));
        return;
    }
    context_.dispatch(
        tall ? linear_coop_tall_fp16_
            : cooperative ? linear_coop_fp16_ : linear_vec16_fp16_,
        {&output, &half_input, &half_weight, &bias},
        &parameters, sizeof(parameters),
        divide_up(output_columns, cooperative ? 16 : 64),
        divide_up(rows, tall ? 64 : cooperative ? 16 : 64));
}

void VulkanOperators::linear_fp16_half_input_output_gelu(
    VulkanBuffer& half_output,
    const VulkanBuffer& half_input,
    const VulkanBuffer& half_weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool weight_transposed) {
    if (!context_.compute_capabilities().fp16 || rows == 0 ||
        input_columns == 0 || output_columns == 0 ||
        input_columns % 4 != 0 || output_columns % 4 != 0) {
        throw std::invalid_argument(
            "invalid FP16 half-input/output linear dimensions");
    }
    require_half_elements(
        half_input, std::uint64_t(rows) * input_columns, "input");
    require_half_elements(
        half_weight,
        std::uint64_t(output_columns) * input_columns,
        "weight");
    require_bytes(bias, output_columns, "bias");
    require_half_elements(
        half_output, std::uint64_t(rows) * output_columns, "output");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
    } parameters{rows, input_columns, output_columns};
    const bool cooperative =
        context_.compute_capabilities().cooperative_matrix_fp16;
    const bool workgroup_nv = rows >= 128 && context_.compute_capabilities()
        .cooperative_matrix_workgroup_epilogue_nv;
    context_.dispatch(
        workgroup_nv
            ? (weight_transposed
                ? linear_coop_workgroup_nv_fp16_transposed_output_gelu_
                : linear_coop_workgroup_nv_fp16_output_gelu_)
            : cooperative
            ? linear_coop_fp16_output_gelu_
            : linear_vec16_fp16_output_gelu_,
        {&half_output, &half_input, &half_weight, &bias},
        &parameters, sizeof(parameters),
        divide_up(output_columns, workgroup_nv ? 32 : cooperative ? 16 : 64),
        divide_up(rows, workgroup_nv ? 32 : cooperative ? 16 : 64));
}

void VulkanOperators::linear_int8(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& int8_weight,
    const VulkanBuffer& weight_scales,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool gelu,
    bool weight_transposed) {
    if (!context_.compute_capabilities().packed_int8_dot) {
        throw std::runtime_error(
            "accelerated INT8 linear operation is unavailable on this Vulkan device");
    }
    if (rows == 0 || input_columns == 0 || output_columns == 0 ||
        input_columns % 4 != 0) {
        throw std::invalid_argument("invalid INT8 linear dimensions");
    }
    const std::uint64_t input_elements =
        std::uint64_t(rows) * input_columns;
    require_bytes(input, input_elements, "input");
    const std::uint64_t packed_weight_words =
        std::uint64_t(output_columns) * (input_columns / 4);
    if (int8_weight.size() < packed_weight_words * sizeof(std::uint32_t)) {
        throw std::invalid_argument("packed INT8 weight buffer is too small");
    }
    require_bytes(weight_scales, output_columns, "weight scale");
    require_bytes(bias, output_columns, "bias");
    require_bytes(output, std::uint64_t(rows) * output_columns, "output");
    VulkanBuffer& input_scales = int8_workspace_.scales(
        std::uint64_t(rows) * sizeof(float),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    VulkanBuffer& packed_input = int8_workspace_.packed(
        std::uint64_t(rows) * (input_columns / 4) * sizeof(std::uint32_t),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    struct QuantizeParameters {
        std::uint32_t rows;
        std::uint32_t columns;
        std::uint32_t input_offset;
    } quantize_parameters{rows, input_columns, 0u};
    context_.dispatch(
        quantize_rows_int8_fused_,
        {&packed_input, &input, &input_scales},
        &quantize_parameters, sizeof(quantize_parameters), rows);
    if (weight_transposed &&
        context_.compute_capabilities().cooperative_matrix_workgroup_int8_nv &&
        rows >= 128) {
        struct Parameters {
            std::uint32_t rows;
            std::uint32_t input_columns;
            std::uint32_t output_columns;
        } parameters{rows, input_columns, output_columns};
        context_.dispatch(
            linear_coop_workgroup_nv_int8_transposed_,
            {&output, &packed_input, &int8_weight, &input_scales,
             &weight_scales, &bias},
            &parameters, sizeof(parameters),
            divide_up(output_columns, 32), divide_up(rows, 32));
    } else {
        linear_int8_packed(
            output, packed_input, int8_weight, input_scales, weight_scales,
            bias, rows, input_columns, output_columns, 0u, rows, false, true);
    }
    if (gelu) {
        struct GeluParameters { std::uint32_t count; } gelu_parameters{
            rows * output_columns};
        context_.dispatch(
            gelu_, {&output, &output}, &gelu_parameters,
            sizeof(gelu_parameters), divide_up(gelu_parameters.count, 256));
    }
}

void VulkanOperators::linear_int8_packed(
    VulkanBuffer& output,
    const VulkanBuffer& packed_input,
    const VulkanBuffer& int8_weight,
    const VulkanBuffer& input_scales,
    const VulkanBuffer& weight_scales,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    std::uint32_t output_row_offset,
    std::uint32_t output_row_stride,
    bool output_transposed,
    bool has_bias) {
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
        std::uint32_t output_row_offset;
        std::uint32_t output_row_stride;
        std::uint32_t output_transposed;
        std::uint32_t has_bias;
    } parameters{
        rows, input_columns, output_columns, output_row_offset,
        output_row_stride, output_transposed ? 1u : 0u,
        has_bias ? 1u : 0u};
    context_.dispatch(
        linear_int8_tiled_,
        {&output, &packed_input, &int8_weight, &input_scales,
         &weight_scales, &bias},
        &parameters, sizeof(parameters),
        divide_up(output_columns, 64),
        divide_up(rows, 56));
}

void VulkanOperators::linear(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool gelu,
    bool block16,
    bool half_weight,
    bool vectorized,
    std::uint32_t vector_tile) {
    if (rows == 0 || input_columns == 0 || output_columns == 0) {
        throw std::invalid_argument("linear dimensions cannot be zero");
    }
    require_bytes(input, std::uint64_t(rows) * input_columns, "input");
    const std::uint64_t weight_elements =
        std::uint64_t(output_columns) * input_columns;
    if (half_weight) {
        require_half_elements(weight, weight_elements, "weight");
    } else {
        require_bytes(weight, weight_elements, "weight");
    }
    require_bytes(bias, output_columns, "bias");
    require_bytes(
        output, std::uint64_t(rows) * output_columns, "output");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
    } parameters{rows, input_columns, output_columns};
    const bool fused_gelu =
        gelu && vectorized && vector_tile == 16 && !half_weight;
    VulkanPipeline& pipeline = fused_gelu
        ? linear_vec16_gelu_
        :
        vectorized
        ? (vector_tile == 16
            ? (half_weight ? linear_vec16_half_ : linear_vec16_)
            : (vector_tile == 8
                ? (half_weight ? linear_vec8_half_ : linear_vec8_)
                : (half_weight ? linear_vec4_half_ : linear_vec4_)))
        : (half_weight
            ? (block16 ? linear16_half_ : linear_half_)
            : (block16 ? linear16_ : linear_));
    const std::uint32_t groups_x = vectorized
        ? divide_up(divide_up(output_columns, 4), 16)
        : divide_up(divide_up(output_columns, 4), 8);
    const std::uint32_t groups_y = vectorized
        ? (vector_tile == 16
            ? divide_up(rows, 64)
            : divide_up(divide_up(rows, 7), 8))
        : divide_up(divide_up(rows, 4), 8);
    context_.dispatch(
        pipeline,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        groups_x,
        groups_y);
    if (gelu && !fused_gelu) {
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

void VulkanOperators::linear_residual_wide(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    const VulkanBuffer& residual,
    const VulkanBuffer& scale,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    std::uint32_t vector_tile,
    bool half_weight) {
    if (rows == 0 || input_columns == 0 || output_columns == 0) {
        throw std::invalid_argument(
            "linear-residual dimensions cannot be zero");
    }
    require_bytes(input, std::uint64_t(rows) * input_columns, "input");
    const std::uint64_t weight_elements =
        std::uint64_t(output_columns) * input_columns;
    if (half_weight) {
        require_half_elements(weight, weight_elements, "weight");
    } else {
        require_bytes(weight, weight_elements, "weight");
    }
    require_bytes(bias, output_columns, "bias");
    require_bytes(
        residual,
        std::uint64_t(rows) * output_columns,
        "residual");
    require_bytes(scale, output_columns, "scale");
    require_bytes(
        output,
        std::uint64_t(rows) * output_columns,
        "output");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
    } parameters{rows, input_columns, output_columns};
    VulkanPipeline* pipeline = nullptr;
    if (vector_tile == 16) {
        pipeline = half_weight
            ? &linear_vec16_half_residual_
            : &linear_vec16_residual_;
    } else {
        pipeline = half_weight
            ? &linear_vec8_half_residual_
            : &linear_vec8_residual_;
    }
    context_.dispatch(
        *pipeline,
        {&output, &input, &weight, &bias, &residual, &scale},
        &parameters,
        sizeof(parameters),
        divide_up(output_columns, 64),
        vector_tile == 16
            ? divide_up(rows, 64)
            : divide_up(divide_up(rows, 7), 8));
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

void VulkanOperators::layer_norm_fp16(
    VulkanBuffer& half_output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t columns,
    float epsilon) {
    if (!context_.compute_capabilities().fp16 || rows == 0 ||
        columns == 0 || epsilon <= 0.0f) {
        throw std::invalid_argument("invalid FP16 layer norm parameters");
    }
    require_bytes(input, std::uint64_t(rows) * columns, "input");
    require_half_elements(
        half_output, std::uint64_t(rows) * columns, "output");
    require_bytes(weight, columns, "weight");
    require_bytes(bias, columns, "bias");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t columns;
        float epsilon;
    } parameters{rows, columns, epsilon};
    context_.dispatch(
        layer_norm_fp16_,
        {&half_output, &input, &weight, &bias},
        &parameters, sizeof(parameters), rows);
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
        {&output, &addend, &scale, &residual},
        &parameters,
        sizeof(parameters),
        divide_up(count, 256));
}

void VulkanOperators::attention_head64(
    VulkanBuffer& output,
    const VulkanBuffer& qkv,
    std::uint32_t tokens,
    std::uint32_t heads,
    VulkanBuffer* score_scratch,
    inferbridge::native::Precision precision) {
    if (tokens == 0 || heads == 0) {
        throw std::invalid_argument("invalid attention dimensions");
    }
    const std::uint64_t elements =
        std::uint64_t(tokens) * heads * 64;
    require_bytes(output, elements, "attention output");
    require_bytes(qkv, elements * 3, "QKV");
    const std::uint64_t score_elements =
        std::uint64_t(heads) * tokens * tokens;
    const bool fp16 = precision == inferbridge::native::Precision::fp16;
    const std::uint64_t score_bytes = fp16
        ? std::uint64_t(heads) * tokens *
            ((std::uint64_t(tokens) + 1) / 2) *
            sizeof(std::uint32_t)
        : score_elements * sizeof(float);
    VulkanBuffer owned_scores;
    if (score_scratch == nullptr) {
        owned_scores = context_.create_device_buffer(score_bytes);
        score_scratch = &owned_scores;
    } else if (score_scratch->size() < score_bytes) {
        throw std::invalid_argument(
            "attention score scratch Vulkan buffer is too small");
    } else if (!fp16) {
        require_bytes(
            *score_scratch, score_elements, "attention score scratch");
    }
    VulkanBuffer& scores = *score_scratch;
    if (fp16) {
        struct HalfParameters {
            std::uint32_t tokens;
            std::uint32_t heads;
        } parameters{tokens, heads};
        const bool cooperative =
            context_.compute_capabilities().cooperative_matrix_fp16;
        context_.dispatch(
            cooperative ? bmm_score_coop_fp16_ : bmm_score_fp16_,
            {&scores, &qkv},
            &parameters,
            sizeof(parameters),
            cooperative ? divide_up(tokens, 16)
                : divide_up(divide_up(tokens, 4), 8),
            cooperative ? divide_up(tokens, 16)
                : divide_up(divide_up(tokens, 8), 8),
            heads);
        struct SoftmaxParameters {
            std::uint32_t rows;
            std::uint32_t columns;
        } softmax_parameters{heads * tokens, tokens};
        context_.dispatch(
            softmax_lastdim_half_,
            {&scores},
            &softmax_parameters,
            sizeof(softmax_parameters),
            softmax_parameters.rows);
        context_.dispatch(
            bmm_value_fp16_,
            {&output, &scores, &qkv},
            &parameters,
            sizeof(parameters),
            divide_up(divide_up(64, 4), 8),
            divide_up(divide_up(tokens, 8), 8),
            heads);
        return;
    }
    struct BmmParameters {
        std::uint32_t rows;
        std::uint32_t columns;
        std::uint32_t inner;
        std::uint32_t batches;
        std::uint32_t weight_transposed;
        std::uint32_t output_token_major;
        std::uint32_t qkv_embedding;
        std::uint32_t input_qkv_query;
        std::uint32_t weight_qkv_kind;
    } score_parameters{
        tokens, tokens, 64, heads, 0, 0, heads * 64, 1, 1};
    context_.dispatch(
        bmm_,
        {&scores, &qkv, &qkv},
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
        {&scores, &scores},
        &softmax_parameters,
        sizeof(softmax_parameters),
        softmax_parameters.rows);
    BmmParameters value_parameters{
        tokens, 64, tokens, heads, 0, 1, heads * 64, 0, 2};
    context_.dispatch(
        bmm_,
        {&output, &scores, &qkv},
        &value_parameters,
        sizeof(value_parameters),
        divide_up(divide_up(64, 4), 8),
        divide_up(divide_up(tokens, 8), 8),
        heads);
}

void VulkanOperators::prepare_tokens(
    VulkanBuffer& output,
    const VulkanBuffer& image,
    const VulkanBuffer& patch_weight,
    const VulkanBuffer& patch_half_weight,
    const VulkanBuffer& patch_int8_weight,
    const VulkanBuffer& patch_int8_scales,
    const VulkanBuffer& patch_bias,
    const VulkanBuffer& class_token,
    const VulkanBuffer& position,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t embedding,
    inferbridge::native::Precision precision) {
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
    const std::uint64_t patch_weight_elements =
        std::uint64_t(embedding) * 3 * 14 * 14;
    if (precision == inferbridge::native::Precision::fp16) {
        require_half_elements(
            patch_half_weight, patch_weight_elements, "patch weight");
    } else if (precision == inferbridge::native::Precision::int8) {
        const std::uint64_t words = patch_weight_elements / 4u;
        if (patch_int8_weight.size() < words * sizeof(std::uint32_t)) {
            throw std::invalid_argument("INT8 patch weight buffer is too small");
        }
        require_bytes(patch_int8_scales, embedding, "patch weight scale");
    } else {
        require_bytes(patch_weight, patch_weight_elements, "patch weight");
    }
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
    if (precision == inferbridge::native::Precision::int8) {
        const std::uint32_t spatial = patch_width * patch_height;
        constexpr std::uint32_t patch_columns = 3u * 14u * 14u;
        VulkanBuffer& packed_input = int8_workspace_.packed(
            std::uint64_t(spatial) * (patch_columns / 4u) *
                sizeof(std::uint32_t),
            [this](std::uint64_t bytes) {
                return context_.create_device_buffer(bytes);
            });
        VulkanBuffer& input_scales = int8_workspace_.scales(
            std::uint64_t(spatial) * sizeof(float),
            [this](std::uint64_t bytes) {
                return context_.create_device_buffer(bytes);
            });
        struct Im2colParameters {
            std::uint32_t input_width;
            std::uint32_t input_height;
            std::uint32_t input_channels;
            std::uint32_t output_width;
            std::uint32_t output_height;
            std::uint32_t kernel;
            std::uint32_t stride;
            std::int32_t padding;
        } im2col_parameters{
            input_width, input_height, 3u, patch_width, patch_height,
            14u, 14u, 0};
        context_.dispatch(
            im2col_quantize_int8_, {&image, &packed_input, &input_scales},
            &im2col_parameters, sizeof(im2col_parameters), spatial);
        struct ClassParameters { std::uint32_t embedding; } class_parameters{
            embedding};
        context_.dispatch(
            copy_class_token_, {&output, &class_token},
            &class_parameters, sizeof(class_parameters),
            divide_up(embedding, 256));
        linear_int8_packed(
            output, packed_input, patch_int8_weight, input_scales,
            patch_int8_scales, patch_bias, spatial, patch_columns,
            embedding, 1u, static_cast<std::uint32_t>(tokens), false, true);
    } else {
        const VulkanBuffer& selected_weight =
            precision == inferbridge::native::Precision::fp16
            ? patch_half_weight : patch_weight;
        VulkanPipeline& pipeline =
            precision == inferbridge::native::Precision::fp16
            ? prepare_tokens_fp16_ : prepare_tokens_;
        context_.dispatch(
            pipeline,
            {&output, &image, &selected_weight, &patch_bias, &class_token},
            &parameters, sizeof(parameters),
            divide_up(embedding, 8),
            divide_up(static_cast<std::uint32_t>(tokens), 8));
    }
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
            patch_width == 37 && patch_height == 37
                ? 1.0f
                : static_cast<float>(
                    1.0 /
                    ((static_cast<double>(patch_width) + 0.1) / 37.0)),
            patch_width == 37 && patch_height == 37
                ? 1.0f
                : static_cast<float>(
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
    context_.write_host(
        output_metadata_buffer, &output_metadata, sizeof(output_metadata));
    context_.write_host(
        input_metadata_buffer, &input_metadata, sizeof(input_metadata));
    context_.write_host(
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
    std::uint32_t output_channels,
    bool half_weight) {
    if (width == 0 || height == 0 || embedding == 0 ||
        output_channels == 0) {
        throw std::invalid_argument("invalid token projection dimensions");
    }
    require_bytes(
        tokens,
        (std::uint64_t(width) * height + 1) * embedding,
        "tokens");
    const std::uint64_t weight_elements =
        std::uint64_t(output_channels) * embedding;
    if (half_weight) {
        require_half_elements(weight, weight_elements, "weight");
    } else {
        require_bytes(weight, weight_elements, "weight");
    }
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
        half_weight ? project_tokens_half_ : project_tokens_,
        {&output, &tokens, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(output_channels, 32),
        divide_up(width * height, 32));
}

void VulkanOperators::project_tokens_fp16(
    VulkanBuffer& output,
    const VulkanBuffer& tokens,
    const VulkanBuffer& half_weight,
    const VulkanBuffer& bias,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t embedding,
    std::uint32_t output_channels) {
    if (!context_.compute_capabilities().fp16) {
        throw std::runtime_error("native FP16 token projection is unavailable");
    }
    const std::uint32_t spatial = width * height;
    require_bytes(tokens, std::uint64_t(spatial + 1u) * embedding, "tokens");
    require_half_elements(
        half_weight, std::uint64_t(output_channels) * embedding, "weight");
    require_bytes(bias, output_channels, "bias");
    require_bytes(output, std::uint64_t(spatial) * output_channels, "output");
    struct Parameters {
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t embedding;
        std::uint32_t output_channels;
    } parameters{width, height, embedding, output_channels};
    context_.dispatch(
        project_tokens_fp16_, {&output, &tokens, &half_weight, &bias},
        &parameters, sizeof(parameters),
        divide_up(output_channels, 32), divide_up(spatial, 32));
}

void VulkanOperators::project_tokens_int8(
    VulkanBuffer& output,
    const VulkanBuffer& tokens,
    const VulkanBuffer& int8_weight,
    const VulkanBuffer& weight_scales,
    const VulkanBuffer& bias,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t embedding,
    std::uint32_t output_channels) {
    if (!context_.compute_capabilities().packed_int8_dot) {
        throw std::runtime_error("native INT8 token projection is unavailable");
    }
    const std::uint32_t spatial = width * height;
    if (embedding % 4u != 0u) {
        throw std::invalid_argument("INT8 token embedding must be divisible by four");
    }
    require_bytes(tokens, std::uint64_t(spatial + 1u) * embedding, "tokens");
    require_bytes(weight_scales, output_channels, "weight scale");
    require_bytes(bias, output_channels, "bias");
    require_bytes(output, std::uint64_t(spatial) * output_channels, "output");
    VulkanBuffer& input_scales = int8_workspace_.scales(
        std::uint64_t(spatial) * sizeof(float),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    VulkanBuffer& packed_input = int8_workspace_.packed(
        std::uint64_t(spatial) * (embedding / 4u) * sizeof(std::uint32_t),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    struct QuantizeParameters {
        std::uint32_t rows;
        std::uint32_t columns;
        std::uint32_t input_offset;
    } quantize_parameters{spatial, embedding, embedding};
    context_.dispatch(
        quantize_rows_int8_fused_,
        {&packed_input, &tokens, &input_scales},
        &quantize_parameters, sizeof(quantize_parameters), spatial);
    linear_int8_packed(
        output, packed_input, int8_weight, input_scales, weight_scales,
        bias, spatial, embedding, output_channels, 0u, spatial, true, true);
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
    bool has_bias,
    bool block8,
    bool half_weight,
    bool tiled,
    bool allow_stride2_tiled) {
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
    const std::uint64_t weight_elements =
        std::uint64_t(output_channels) * input_channels * kernel * kernel;
    if (half_weight) {
        require_half_elements(
            weight, weight_elements, "convolution weight");
    } else {
        require_bytes(weight, weight_elements, "convolution weight");
    }
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
    const bool use_tiled =
        tiled && kernel == 3 && stride == 1 && padding == 1 &&
        output_width == input_width && output_height == input_height;
    const char* disable_half_specialized =
        std::getenv("DAV2_DISABLE_FP16_SPECIALIZED_CONV");
    const bool allow_half_specialized =
        disable_half_specialized == nullptr ||
        disable_half_specialized[0] == '\0' ||
        disable_half_specialized[0] == '0';
    const bool specialized_weight = !half_weight || allow_half_specialized;
    const bool use_tiled16x8 = use_tiled && !block8 && specialized_weight;
    const bool pointwise =
        specialized_weight && kernel == 1 && stride == 1 && padding == 0 &&
        output_width == input_width && output_height == input_height;
    const char* disable_pointwise_matrix2 =
        std::getenv("DAV2_DISABLE_MATRIX2_POINTWISE_CONV");
    const bool pointwise_matrix2 = pointwise && half_weight &&
        context_.compute_capabilities().cooperative_matrix_workgroup_nv &&
        std::uint64_t(output_width) * output_height >= 128 &&
        (disable_pointwise_matrix2 == nullptr ||
         disable_pointwise_matrix2[0] == '\0' ||
         disable_pointwise_matrix2[0] == '0');
    const bool stride2_tiled =
        specialized_weight && allow_stride2_tiled && kernel == 3 &&
        stride == 2 && padding == 1 && input_channels <= 384 &&
        output_channels <= 384;
    if (pointwise_matrix2) {
        const std::uint32_t rows = output_width * output_height;
        VulkanBuffer& packed_input = fp16_workspace_.ensure(
            std::uint64_t(rows) * input_channels * sizeof(std::uint16_t),
            [this](std::uint64_t bytes) {
                return context_.create_device_buffer(bytes);
            });
        struct PackParameters { std::uint32_t count; } pack_parameters{
            rows * input_channels};
        context_.dispatch(
            pack_fp16_, {&packed_input, &input},
            &pack_parameters, sizeof(pack_parameters),
            divide_up(pack_parameters.count, 256));
        struct MatrixParameters {
            std::uint32_t rows;
            std::uint32_t input_columns;
            std::uint32_t output_columns;
        } matrix_parameters{rows, input_channels, output_channels};
        context_.dispatch(
            conv2d_pointwise_coop_workgroup_nv_fp16_,
            {&output, &packed_input, &weight, &bias},
            &matrix_parameters, sizeof(matrix_parameters),
            divide_up(output_channels, 32), divide_up(rows, 32));
        return;
    }
    VulkanPipeline& pipeline =
        pointwise
        ? (half_weight
            ? conv2d_pointwise_gemm_half_
            : conv2d_pointwise_gemm_)
        : stride2_tiled
        ? (half_weight
            ? conv2d_stride2_tiled8x8_half_
            : conv2d_stride2_tiled8x8_)
        : (use_tiled16x8
        ? (half_weight ? conv2d_tiled16x8_half_ : conv2d_tiled16x8_)
        : (use_tiled
        ? (half_weight
            ? (block8 ? conv2d8_tiled_half_ : conv2d_tiled_half_)
            : (block8 ? conv2d8_tiled_ : conv2d_tiled_))
        : (half_weight
            ? (block8 ? conv2d8_half_ : conv2d_half_)
            : (block8 ? conv2d8_ : conv2d_))));
    context_.dispatch(
        pipeline,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        pointwise ? divide_up(output_width * output_height, 32)
            : divide_up(output_width, use_tiled16x8 ? 16 : 8),
        pointwise ? divide_up(output_channels, 32)
            : divide_up(output_height, 8),
        pointwise ? 1
            : stride2_tiled
                ? divide_up(output_channels, 8)
            : divide_up(
                output_channels,
                (block8 || use_tiled16x8) ? 8 : 4));
}

void VulkanOperators::conv2d_fp16(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& half_weight,
    const VulkanBuffer& bias,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t input_channels,
    std::uint32_t output_channels,
    std::uint32_t kernel,
    std::uint32_t stride,
    std::uint32_t padding,
    bool has_bias) {
    if (!context_.compute_capabilities().fp16) {
        throw std::runtime_error("native FP16 convolution is unavailable");
    }
    if (input_width == 0 || input_height == 0 || input_channels == 0 ||
        output_channels == 0 || kernel == 0 || stride == 0 ||
        input_width + 2u * padding < kernel ||
        input_height + 2u * padding < kernel) {
        throw std::invalid_argument("invalid FP16 convolution dimensions");
    }
    const std::uint32_t output_width =
        (input_width + 2u * padding - kernel) / stride + 1u;
    const std::uint32_t output_height =
        (input_height + 2u * padding - kernel) / stride + 1u;
    require_bytes(
        input, std::uint64_t(input_width) * input_height * input_channels,
        "convolution input");
    require_half_elements(
        half_weight,
        std::uint64_t(output_channels) * input_channels * kernel * kernel,
        "convolution weight");
    require_bytes(bias, has_bias ? output_channels : 1u, "convolution bias");
    require_bytes(
        output, std::uint64_t(output_width) * output_height * output_channels,
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
        input_width, input_height, input_channels, output_width, output_height,
        output_channels, kernel, stride, static_cast<std::int32_t>(padding),
        has_bias ? 1u : 0u};
    context_.dispatch(
        conv2d_fp16_, {&output, &input, &half_weight, &bias},
        &parameters, sizeof(parameters), divide_up(output_width, 8),
        divide_up(output_height, 8), divide_up(output_channels, 4));
}

void VulkanOperators::conv2d_int8(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& int8_weight,
    const VulkanBuffer& weight_scales,
    const VulkanBuffer& bias,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t input_channels,
    std::uint32_t output_channels,
    std::uint32_t kernel,
    std::uint32_t stride,
    std::uint32_t padding,
    bool has_bias) {
    if (!context_.compute_capabilities().packed_int8_dot) {
        throw std::runtime_error("native INT8 convolution is unavailable");
    }
    if (input_width == 0 || input_height == 0 || input_channels == 0 ||
        output_channels == 0 || kernel == 0 || stride == 0 ||
        input_width + 2u * padding < kernel ||
        input_height + 2u * padding < kernel) {
        throw std::invalid_argument("invalid INT8 convolution dimensions");
    }
    const std::uint32_t inner = input_channels * kernel * kernel;
    if (inner % 4u != 0u) {
        throw std::invalid_argument("INT8 convolution inner size is not packed");
    }
    const std::uint32_t output_width =
        (input_width + 2u * padding - kernel) / stride + 1u;
    const std::uint32_t output_height =
        (input_height + 2u * padding - kernel) / stride + 1u;
    const std::uint32_t rows = output_width * output_height;
    require_bytes(
        input, std::uint64_t(input_width) * input_height * input_channels,
        "convolution input");
    if (int8_weight.size() <
        std::uint64_t(output_channels) * (inner / 4u) * sizeof(std::uint32_t)) {
        throw std::invalid_argument("INT8 convolution weight is too small");
    }
    require_bytes(weight_scales, output_channels, "convolution weight scale");
    require_bytes(bias, has_bias ? output_channels : 1u, "convolution bias");
    require_bytes(output, std::uint64_t(rows) * output_channels, "output");
    VulkanBuffer& packed_input = int8_workspace_.packed(
        std::uint64_t(rows) * (inner / 4u) * sizeof(std::uint32_t),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    VulkanBuffer& input_scales = int8_workspace_.scales(
        std::uint64_t(rows) * sizeof(float),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    struct Im2colParameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_width;
        std::uint32_t output_height;
        std::uint32_t kernel;
        std::uint32_t stride;
        std::int32_t padding;
    } parameters{
        input_width, input_height, input_channels, output_width, output_height,
        kernel, stride, static_cast<std::int32_t>(padding)};
    context_.dispatch(
        im2col_quantize_int8_, {&input, &packed_input, &input_scales},
        &parameters, sizeof(parameters), rows);
    linear_int8_packed(
        output, packed_input, int8_weight, input_scales, weight_scales,
        bias, rows, inner, output_channels, 0u, rows, true, has_bias);
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
    std::uint32_t kernel,
    bool half_weight) {
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
    const std::uint64_t weight_elements =
        std::uint64_t(input_channels) * output_channels * kernel * kernel;
    if (half_weight) {
        require_half_elements(
            weight, weight_elements, "transposed convolution weight");
    } else {
        require_bytes(
            weight, weight_elements, "transposed convolution weight");
    }
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
        half_weight
            ? conv_transpose_nonoverlap_half_
            : conv_transpose_nonoverlap_,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8),
        output_channels);
}

void VulkanOperators::conv_transpose_nonoverlap_fp16(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& half_weight,
    const VulkanBuffer& bias,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t input_channels,
    std::uint32_t output_channels,
    std::uint32_t kernel) {
    if (!context_.compute_capabilities().fp16) {
        throw std::runtime_error(
            "native FP16 transposed convolution is unavailable");
    }
    if (input_width == 0 || input_height == 0 || input_channels == 0 ||
        output_channels == 0 || kernel == 0) {
        throw std::invalid_argument("invalid FP16 transposed convolution");
    }
    const std::uint32_t output_width = input_width * kernel;
    const std::uint32_t output_height = input_height * kernel;
    require_bytes(
        input, std::uint64_t(input_width) * input_height * input_channels,
        "transposed convolution input");
    require_half_elements(
        half_weight,
        std::uint64_t(input_channels) * output_channels * kernel * kernel,
        "transposed convolution weight");
    require_bytes(bias, output_channels, "transposed convolution bias");
    require_bytes(
        output, std::uint64_t(output_width) * output_height * output_channels,
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
        conv_transpose_nonoverlap_fp16_,
        {&output, &input, &half_weight, &bias},
        &parameters, sizeof(parameters), divide_up(output_width, 8),
        divide_up(output_height, 8), output_channels);
}

void VulkanOperators::conv_transpose_nonoverlap_int8(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& int8_weight,
    const VulkanBuffer& weight_scales,
    const VulkanBuffer& bias,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t input_channels,
    std::uint32_t output_channels,
    std::uint32_t kernel) {
    if (!context_.compute_capabilities().packed_int8_dot) {
        throw std::runtime_error(
            "native INT8 transposed convolution is unavailable");
    }
    if (input_width == 0 || input_height == 0 || input_channels == 0 ||
        output_channels == 0 || kernel == 0 || input_channels % 4u != 0u) {
        throw std::invalid_argument("invalid INT8 transposed convolution");
    }
    const std::uint32_t input_rows = input_width * input_height;
    const std::uint32_t packed_columns = input_channels / 4u;
    const std::uint32_t weight_rows = kernel * kernel * output_channels;
    if (int8_weight.size() <
        std::uint64_t(weight_rows) * packed_columns * sizeof(std::uint32_t)) {
        throw std::invalid_argument("INT8 transposed weight is too small");
    }
    require_bytes(weight_scales, weight_rows, "transposed weight scale");
    require_bytes(bias, output_channels, "transposed bias");
    VulkanBuffer& packed_input = int8_workspace_.packed(
        std::uint64_t(input_rows) * packed_columns * sizeof(std::uint32_t),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    VulkanBuffer& input_scales = int8_workspace_.scales(
        std::uint64_t(input_rows) * sizeof(float),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    struct Im2colParameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_width;
        std::uint32_t output_height;
        std::uint32_t kernel;
        std::uint32_t stride;
        std::int32_t padding;
    } im2col_parameters{
        input_width, input_height, input_channels, input_width, input_height,
        1u, 1u, 0};
    context_.dispatch(
        im2col_quantize_int8_, {&input, &packed_input, &input_scales},
        &im2col_parameters, sizeof(im2col_parameters), input_rows);
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_channels;
        std::uint32_t kernel;
    } parameters{
        input_width, input_height, input_channels, output_channels, kernel};
    context_.dispatch(
        conv_transpose_int8_,
        {&output, &packed_input, &int8_weight, &input_scales,
         &weight_scales, &bias},
        &parameters, sizeof(parameters),
        divide_up(input_width * kernel, 8),
        divide_up(input_height * kernel, 8), output_channels);
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

void VulkanOperators::bilinear_align_true_image(
    VulkanImage& output,
    const VulkanBuffer& input,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t output_width,
    std::uint32_t output_height) {
    if (input_width == 0 || input_height == 0 ||
        output_width == 0 || output_height == 0 ||
        output.width() != output_width ||
        output.height() != output_height ||
        output.format() != VK_FORMAT_R32_SFLOAT) {
        throw std::invalid_argument(
            "invalid bilinear image dimensions or format");
    }
    require_bytes(
        input,
        std::uint64_t(input_width) * input_height,
        "bilinear image input");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t output_width;
        std::uint32_t output_height;
    } parameters{
        input_width, input_height, output_width, output_height};
    context_.dispatch_buffer_to_image(
        bilinear_align_true_image_,
        input,
        output,
        &parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8));
}

void VulkanOperators::reduce_minmax(
    const VulkanBuffer& input,
    VulkanBuffer& range,
    std::uint32_t count) {
    if (count == 0) throw std::invalid_argument("empty depth reduction");
    require_bytes(input, count, "depth reduction input");
    require_bytes(range, 2, "depth reduction range");
    context_.dispatch(
        reduce_minmax_, {&input, &range}, &count, sizeof(count), 1);
}

void VulkanOperators::normalize_relative(
    VulkanBuffer& depth,
    const VulkanBuffer& range,
    std::uint32_t count) {
    if (count == 0) throw std::invalid_argument("empty relative depth");
    require_bytes(depth, count, "relative depth");
    require_bytes(range, 2, "relative depth range");
    context_.dispatch(
        normalize_relative_, {&depth, &range}, &count, sizeof(count),
        divide_up(count, 256));
}

void VulkanOperators::normalize_metric(
    VulkanBuffer& depth,
    const VulkanBuffer& range,
    std::uint32_t count,
    float maximum_depth) {
    if (count == 0 || !(maximum_depth > 0.0f)) {
        throw std::invalid_argument("invalid metric depth normalization");
    }
    require_bytes(depth, count, "metric depth");
    require_bytes(range, 2, "metric depth range");
    struct Parameters {
        std::uint32_t count;
        float maximum_depth;
    } parameters{count, maximum_depth};
    context_.dispatch(
        normalize_metric_, {&depth, &range},
        &parameters, sizeof(parameters), divide_up(count, 256));
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

void VulkanOperators::sigmoid_scale(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    std::uint32_t count,
    float scale) {
    require_bytes(output, count, "sigmoid output");
    require_bytes(input, count, "sigmoid input");
    struct Parameters {
        std::uint32_t count;
        float scale;
    } parameters{count, scale};
    context_.dispatch(
        sigmoid_scale_, {&output, &input},
        &parameters, sizeof(parameters), divide_up(count, 256));
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
