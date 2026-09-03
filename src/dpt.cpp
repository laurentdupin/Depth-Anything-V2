#include "dpt.h"
#include "inferbridge/native_harness_precision.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace dav2 {
namespace {

std::uint64_t elements(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t channels) {
    return std::uint64_t(width) * height * channels;
}

const VulkanBuffer& weight(
    const GpuModel& model,
    const std::string& name) {
    return model.tensor(name).buffer;
}

}  // namespace

DptHead::DptHead(
    dav2_encoder encoder,
    VulkanContext& context,
    GpuModel& weights,
    VulkanOperators& operators,
    inferbridge::native::Precision precision,
    float metric_max_depth)
    : context_(context),
      weights_(weights),
      operators_(operators),
      zero_bias_(context.create_device_buffer(sizeof(float))),
      precision_(precision),
      refinement_precision_(
          precision == inferbridge::native::Precision::int8
          ? (context.compute_capabilities().fp16
              ? inferbridge::native::Precision::fp16
              : inferbridge::native::Precision::fp32)
          : precision),
      metric_max_depth_(metric_max_depth) {
    const float zero = 0.0f;
    context_.upload(zero_bias_, &zero, sizeof(zero));
    switch (encoder) {
        case DAV2_ENCODER_VITS:
            embedding_ = 384;
            features_ = 64;
            project_channels_[0] = 48;
            project_channels_[1] = 96;
            project_channels_[2] = 192;
            project_channels_[3] = 384;
            break;
        case DAV2_ENCODER_VITB:
            embedding_ = 768;
            features_ = 128;
            project_channels_[0] = 96;
            project_channels_[1] = 192;
            project_channels_[2] = 384;
            project_channels_[3] = 768;
            break;
        case DAV2_ENCODER_VITL:
            embedding_ = 1024;
            features_ = 256;
            project_channels_[0] = 256;
            project_channels_[1] = 512;
            project_channels_[2] = 1024;
            project_channels_[3] = 1024;
            break;
        default:
            throw std::invalid_argument("unsupported DPT encoder");
    }
}

void DptHead::select_convolution_block(
    std::uint32_t patch_width,
    std::uint32_t patch_height) {
    constexpr std::uint32_t side = 16;
    const VkDeviceSize bytes =
        elements(side, side, features_) * sizeof(float);
    VulkanBuffer input = context_.create_device_buffer(bytes);
    VulkanBuffer output = context_.create_device_buffer(bytes);
    const GpuTensor& convolution_weight = weights_.tensor(
        "depth_head.scratch.refinenet4.resConfUnit2.conv1.weight");
    const VulkanBuffer& convolution_bias = weight(
        weights_,
        "depth_head.scratch.refinenet4.resConfUnit2.conv1.bias");
    const auto run = [&](bool block8, bool half_weight) {
        const auto start = std::chrono::steady_clock::now();
        context_.batch([&] {
            for (int repetition = 0; repetition < 3; ++repetition) {
                operators_.conv2d(
                    output,
                    input,
                    half_weight
                        ? convolution_weight.half_buffer
                        : convolution_weight.buffer,
                    convolution_bias,
                    side,
                    side,
                    features_,
                    features_,
                    3,
                    1,
                    1,
                    true,
                    block8,
                    half_weight,
                    false);
            }
        });
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };
    struct Candidate {
        bool block8;
        bool half_weight;
        std::array<double, 3> samples{};
    };
    std::array<Candidate, 4> candidates{{
        {false, false, {}},
        {true, false, {}},
        {false, true, {}},
        {true, true, {}},
    }};
    for (Candidate& candidate : candidates) {
        run(candidate.block8, candidate.half_weight);
    }
    for (std::size_t sample = 0;
         sample < candidates[0].samples.size();
         ++sample) {
        if ((sample & 1u) == 0) {
            for (Candidate& candidate : candidates) {
                candidate.samples[sample] =
                    run(candidate.block8, candidate.half_weight);
            }
        } else {
            for (auto candidate = candidates.rbegin();
                 candidate != candidates.rend();
                 ++candidate) {
                candidate->samples[sample] =
                    run(candidate->block8, candidate->half_weight);
            }
        }
    }
    Candidate* best_fp32 = nullptr;
    Candidate* best_half = nullptr;
    double best_fp32_time = 0.0;
    double best_half_time = 0.0;
    for (Candidate& candidate : candidates) {
        std::sort(candidate.samples.begin(), candidate.samples.end());
        const double median =
            candidate.samples[candidate.samples.size() / 2];
        Candidate*& best =
            candidate.half_weight ? best_half : best_fp32;
        double& best_time =
            candidate.half_weight ? best_half_time : best_fp32_time;
        if (best == nullptr || median < best_time) {
            best = &candidate;
            best_time = median;
        }
    }
    Candidate* best =
        refinement_precision_ == inferbridge::native::Precision::fp16
        ? best_half : best_fp32;
    convolution_block8_ = best->block8;
    convolution_half_weight_ = best->half_weight;
    if (refinement_precision_ == inferbridge::native::Precision::fp16) {
        std::array<double, 3> native_samples{};
        const auto run_native = [&] {
            const auto start = std::chrono::steady_clock::now();
            context_.batch([&] {
                for (int repetition = 0; repetition < 3; ++repetition) {
                    operators_.conv2d_fp16(
                        output, input, convolution_weight.half_buffer,
                        convolution_bias, side, side, features_, features_,
                        3, 1, 1, true);
                }
            });
            return std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count();
        };
        run_native();
        for (double& sample : native_samples) sample = run_native();
        std::sort(native_samples.begin(), native_samples.end());
        convolution_native_fp16_ =
            native_samples[1] < best_half_time * 0.985;
    }
    const std::uint32_t full_width = patch_width * 14;
    const std::uint32_t full_height = patch_height * 14;
    VulkanBuffer full_input = context_.create_device_buffer(
        elements(full_width, full_height, features_ / 2) * sizeof(float));
    VulkanBuffer full_output = context_.create_device_buffer(
        elements(full_width, full_height, 32) * sizeof(float));
    const GpuTensor& output_weight = weights_.tensor(
        "depth_head.scratch.output_conv2.0.weight");
    const VulkanBuffer& output_bias = weight(
        weights_,
        "depth_head.scratch.output_conv2.0.bias");
    const std::uint32_t largest_width = patch_width * 4;
    const std::uint32_t largest_height = patch_height * 4;
    const VkDeviceSize refinement_bytes =
        elements(largest_width, largest_height, features_) * sizeof(float);
    VulkanBuffer refinement_input =
        context_.create_device_buffer(refinement_bytes);
    VulkanBuffer refinement_output =
        context_.create_device_buffer(refinement_bytes);
    const GpuTensor& refinement_weight = weights_.tensor(
        "depth_head.scratch.refinenet1.resConfUnit2.conv1.weight");
    const VulkanBuffer& refinement_bias = weight(
        weights_,
        "depth_head.scratch.refinenet1.resConfUnit2.conv1.bias");
    const auto run_convolution_mix = [&](bool block8, bool tiled) {
        const auto start = std::chrono::steady_clock::now();
        context_.batch([&] {
            // Approximate the complete DPT 3x3 mix. The refinement units
            // dominate, while one full-resolution 32-channel convolution
            // captures the depth head's different shape.
            constexpr std::array<std::uint32_t, 4> scale = {1, 1, 2, 4};
            constexpr std::array<std::uint32_t, 4> repetitions = {4, 6, 6, 4};
            for (std::size_t level = 0; level < scale.size(); ++level) {
                const std::uint32_t width =
                    std::max(1u, patch_width * scale[level] / 2);
                const std::uint32_t height =
                    std::max(1u, patch_height * scale[level] / 2);
                for (std::uint32_t repetition = 0;
                     repetition < repetitions[level];
                     ++repetition) {
                    operators_.conv2d(
                        refinement_output,
                        refinement_input,
                        convolution_half_weight_
                            ? refinement_weight.half_buffer
                            : refinement_weight.buffer,
                        refinement_bias,
                        width,
                        height,
                        features_,
                        features_,
                        3,
                        1,
                        1,
                        true,
                        block8,
                        convolution_half_weight_,
                        tiled);
                }
            }
            operators_.conv2d(
                full_output,
                full_input,
                convolution_half_weight_
                    ? output_weight.half_buffer
                    : output_weight.buffer,
                output_bias,
                full_width,
                full_height,
                features_ / 2,
                32,
                3,
                1,
                1,
                true,
                block8,
                convolution_half_weight_,
                tiled);
        });
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };
    struct ConvolutionCandidate {
        bool block8;
        bool tiled;
        std::array<double, 3> samples{};
    };
    std::array<ConvolutionCandidate, 4> convolution_candidates{{
        {false, false, {}},
        {true, false, {}},
        {false, true, {}},
        {true, true, {}},
    }};
    for (const ConvolutionCandidate& candidate : convolution_candidates)
        run_convolution_mix(candidate.block8, candidate.tiled);
    for (std::size_t sample = 0;
         sample < convolution_candidates[0].samples.size(); ++sample) {
        if ((sample & 1u) == 0) {
            for (ConvolutionCandidate& candidate : convolution_candidates) {
                candidate.samples[sample] = run_convolution_mix(
                    candidate.block8, candidate.tiled);
            }
        } else {
            for (auto candidate = convolution_candidates.rbegin();
                 candidate != convolution_candidates.rend(); ++candidate) {
                candidate->samples[sample] = run_convolution_mix(
                    candidate->block8, candidate->tiled);
            }
        }
    }
    ConvolutionCandidate* best_convolution = nullptr;
    double best_convolution_time = 0.0;
    for (ConvolutionCandidate& candidate : convolution_candidates) {
        std::sort(candidate.samples.begin(), candidate.samples.end());
        const double median = candidate.samples[1];
        if (best_convolution == nullptr || median < best_convolution_time) {
            best_convolution = &candidate;
            best_convolution_time = median;
        }
    }
    convolution_block8_ = best_convolution->block8;
    convolution_tiled_ = best_convolution->tiled;
    if (refinement_precision_ == inferbridge::native::Precision::fp16) {
        const auto run_native_mix = [&] {
            const auto start = std::chrono::steady_clock::now();
            context_.batch([&] {
                constexpr std::array<std::uint32_t, 4> scale = {1, 1, 2, 4};
                constexpr std::array<std::uint32_t, 4> repetitions = {4, 6, 6, 4};
                for (std::size_t level = 0; level < scale.size(); ++level) {
                    const std::uint32_t width =
                        std::max(1u, patch_width * scale[level] / 2);
                    const std::uint32_t height =
                        std::max(1u, patch_height * scale[level] / 2);
                    for (std::uint32_t repetition = 0;
                         repetition < repetitions[level]; ++repetition) {
                        operators_.conv2d_fp16(
                            refinement_output, refinement_input,
                            refinement_weight.half_buffer, refinement_bias,
                            width, height, features_, features_, 3, 1, 1,
                            true);
                    }
                }
                operators_.conv2d_fp16(
                    full_output, full_input, output_weight.half_buffer,
                    output_bias, full_width, full_height, features_ / 2, 32,
                    3, 1, 1, true);
            });
            return std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count();
        };
        std::array<double, 3> selected_mix_samples{};
        std::array<double, 3> native_mix_samples{};
        run_convolution_mix(convolution_block8_, convolution_tiled_);
        run_native_mix();
        for (std::size_t sample = 0;
             sample < selected_mix_samples.size(); ++sample) {
            if ((sample & 1u) == 0) {
                selected_mix_samples[sample] =
                    run_convolution_mix(
                        convolution_block8_, convolution_tiled_);
                native_mix_samples[sample] = run_native_mix();
            } else {
                native_mix_samples[sample] = run_native_mix();
                selected_mix_samples[sample] =
                    run_convolution_mix(
                        convolution_block8_, convolution_tiled_);
            }
        }
        std::sort(
            selected_mix_samples.begin(), selected_mix_samples.end());
        std::sort(native_mix_samples.begin(), native_mix_samples.end());
        convolution_native_fp16_ =
            native_mix_samples[1] < selected_mix_samples[1] * 0.985;
        if (std::getenv("DAV2_PRINT_CONVOLUTION_TUNING")) {
            std::fprintf(
                stderr,
                "DAV2_CONV_TUNING:selected_us=%.3f native_us=%.3f native=%d tiled=%d block8=%d\n",
                selected_mix_samples[1], native_mix_samples[1],
                convolution_native_fp16_ ? 1 : 0,
                convolution_tiled_ ? 1 : 0,
                convolution_block8_ ? 1 : 0);
        }
    }
    if (project_channels_[3] <= 384) {
        const std::uint32_t stride_input_width =
            std::max(1u, patch_width);
        const std::uint32_t stride_input_height =
            std::max(1u, patch_height);
        const std::uint32_t stride_output_width =
            (stride_input_width + 1) / 2;
        const std::uint32_t stride_output_height =
            (stride_input_height + 1) / 2;
        VulkanBuffer stride_input = context_.create_device_buffer(
            elements(
                stride_input_width,
                stride_input_height,
                project_channels_[3]) *
            sizeof(float));
        VulkanBuffer stride_output = context_.create_device_buffer(
            elements(
                stride_output_width,
                stride_output_height,
                project_channels_[3]) *
            sizeof(float));
        const GpuTensor& stride_weight = weights_.tensor(
            "depth_head.resize_layers.3.weight");
        const VulkanBuffer& stride_bias = weight(
            weights_, "depth_head.resize_layers.3.bias");
        const auto run_stride2 = [&](bool use_specialized) {
            const auto start = std::chrono::steady_clock::now();
            context_.batch([&] {
                for (int repetition = 0; repetition < 3; ++repetition) {
                    operators_.conv2d(
                        stride_output,
                        stride_input,
                        convolution_half_weight_
                            ? stride_weight.half_buffer
                            : stride_weight.buffer,
                        stride_bias,
                        stride_input_width,
                        stride_input_height,
                        project_channels_[3],
                        project_channels_[3],
                        3,
                        2,
                        1,
                        true,
                        convolution_block8_,
                        convolution_half_weight_,
                        convolution_tiled_,
                        use_specialized);
                }
            });
            return std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count();
        };
        std::array<double, 3> generic_stride_samples{};
        std::array<double, 3> tiled_stride_samples{};
        run_stride2(false);
        run_stride2(true);
        for (std::size_t sample = 0;
             sample < generic_stride_samples.size();
             ++sample) {
            if ((sample & 1u) == 0) {
                generic_stride_samples[sample] = run_stride2(false);
                tiled_stride_samples[sample] = run_stride2(true);
            } else {
                tiled_stride_samples[sample] = run_stride2(true);
                generic_stride_samples[sample] = run_stride2(false);
            }
        }
        std::sort(
            generic_stride_samples.begin(),
            generic_stride_samples.end());
        std::sort(
            tiled_stride_samples.begin(),
            tiled_stride_samples.end());
        stride2_tiled_ =
            tiled_stride_samples[1] < generic_stride_samples[1] * 0.985;
    }
    // Diagnostic overrides make convolution routing reproducible when
    // comparing kernels on a particular adapter. They are intentionally
    // absent from normal execution, which continues to use autotuning.
    if (const char* value = std::getenv("DAV2_CONV_BLOCK8"))
        convolution_block8_ = value[0] != '0';
    if (const char* value = std::getenv("DAV2_CONV_TILED"))
        convolution_tiled_ = value[0] != '0';
    if (const char* value = std::getenv("DAV2_CONV_STRIDE2_TILED"))
        stride2_tiled_ = value[0] != '0';
    weights_.retain_dpt_precision(
        precision_, convolution_half_weight_
            ? inferbridge::native::Precision::fp16
            : inferbridge::native::Precision::fp32);
    convolution_block_selected_ = true;
}

const VulkanBuffer& DptHead::selected_weight(
    const std::string& name) const {
    const GpuTensor& tensor = weights_.tensor(name);
    return convolution_half_weight_
        ? tensor.half_buffer
        : tensor.buffer;
}

FeatureMap DptHead::conv(
    FeatureMap&& input,
    const std::string& weight_name,
    const std::string& bias_name,
    std::uint32_t output_channels,
    std::uint32_t kernel,
    std::uint32_t stride,
    std::uint32_t padding,
    bool has_bias) {
    const std::uint32_t output_width =
        (input.width + 2 * padding - kernel) / stride + 1;
    const std::uint32_t output_height =
        (input.height + 2 * padding - kernel) / stride + 1;
    FeatureMap output{
        context_.create_device_buffer(
            elements(output_width, output_height, output_channels) *
            sizeof(float)),
        output_width,
        output_height,
        output_channels,
    };
    const GpuTensor& convolution_weight =
        weights_.tensor(weight_name);
    const VulkanBuffer& bias =
        has_bias ? weight(weights_, bias_name) : zero_bias_;
    const inferbridge::native::Precision operation_precision =
        weight_name.rfind("depth_head.resize_layers.", 0) == 0
        ? precision_ : refinement_precision_;
    if (operation_precision == inferbridge::native::Precision::fp16) {
        if (convolution_native_fp16_) {
            operators_.conv2d_fp16(
                output.buffer, input.buffer, convolution_weight.half_buffer,
                bias, input.width, input.height, input.channels,
                output_channels, kernel, stride, padding, has_bias);
        } else {
            operators_.conv2d(
                output.buffer, input.buffer, convolution_weight.half_buffer,
                bias, input.width, input.height, input.channels,
                output_channels, kernel, stride, padding, has_bias,
                convolution_block8_, true, convolution_tiled_,
                stride2_tiled_);
        }
    } else if (operation_precision == inferbridge::native::Precision::int8) {
        operators_.conv2d_int8(
            output.buffer, input.buffer, convolution_weight.int8_buffer,
            convolution_weight.int8_scales, bias,
            input.width, input.height, input.channels, output_channels,
            kernel, stride, padding, has_bias);
    } else {
        operators_.conv2d(
            output.buffer, input.buffer, convolution_weight.buffer, bias,
            input.width, input.height, input.channels, output_channels,
            kernel, stride, padding, has_bias, convolution_block8_, false,
            convolution_tiled_, stride2_tiled_);
    }
    return output;
}

FeatureMap DptHead::residual_unit(
    FeatureMap&& input,
    const std::string& prefix) {
    const std::uint32_t count =
        static_cast<std::uint32_t>(
            elements(input.width, input.height, input.channels));
    FeatureMap activated{
        context_.create_device_buffer(std::uint64_t(count) * sizeof(float)),
        input.width,
        input.height,
        input.channels,
    };
    operators_.relu(activated.buffer, input.buffer, count);
    FeatureMap first = conv(
        std::move(activated),
        prefix + ".conv1.weight",
        prefix + ".conv1.bias",
        input.channels,
        3,
        1,
        1,
        true);
    operators_.relu(first.buffer, first.buffer, count);
    FeatureMap second = conv(
        std::move(first),
        prefix + ".conv2.weight",
        prefix + ".conv2.bias",
        input.channels,
        3,
        1,
        1,
        true);
    operators_.add(
        second.buffer, second.buffer, input.buffer, count);
    return second;
}

FeatureMap DptHead::fusion(
    FeatureMap&& path,
    FeatureMap&& skip,
    const std::string& prefix,
    std::uint32_t output_width,
    std::uint32_t output_height) {
    if (skip.buffer.handle() != VK_NULL_HANDLE) {
        FeatureMap processed_skip =
            residual_unit(std::move(skip), prefix + ".resConfUnit1");
        const std::uint32_t count =
            static_cast<std::uint32_t>(
                elements(path.width, path.height, path.channels));
        operators_.add(
            path.buffer, path.buffer, processed_skip.buffer, count);
    }
    path = residual_unit(
        std::move(path), prefix + ".resConfUnit2");
    FeatureMap resized{
        context_.create_device_buffer(
            elements(output_width, output_height, features_) * sizeof(float)),
        output_width,
        output_height,
        features_,
    };
    operators_.bilinear_align_true(
        resized.buffer,
        path.buffer,
        path.width,
        path.height,
        output_width,
        output_height,
        features_);
    return conv(
        std::move(resized),
        prefix + ".out_conv.weight",
        prefix + ".out_conv.bias",
        features_,
        1,
        1,
        0,
        true);
}

void DptHead::prepare(
    std::uint32_t patch_width,
    std::uint32_t patch_height) {
    if (patch_width == 0 || patch_height == 0) {
        throw std::invalid_argument("DPT patch dimensions cannot be zero");
    }
    if (!convolution_block_selected_) {
        select_convolution_block(patch_width, patch_height);
    }
}

FeatureMap DptHead::forward(EncoderOutput&& encoded) {
    if (encoded.features.size() != 4 ||
        encoded.embedding != embedding_) {
        throw std::invalid_argument("invalid DPT encoder output");
    }
    prepare(encoded.patch_width, encoded.patch_height);
    FeatureMap layers[4];
    const auto run_projections = [&] {
        for (std::uint32_t index = 0; index < 4; ++index) {
            context_.batch([&] {
            FeatureMap projected{
                context_.create_device_buffer(
                    elements(
                        encoded.patch_width,
                        encoded.patch_height,
                        project_channels_[index]) *
                    sizeof(float)),
                encoded.patch_width,
                encoded.patch_height,
                project_channels_[index],
            };
            const std::string prefix =
                "depth_head.projects." + std::to_string(index);
            const GpuTensor& projection_weight =
                weights_.tensor(prefix + ".weight");
            const VulkanBuffer& projection_bias =
                weight(weights_, prefix + ".bias");
            if (precision_ == inferbridge::native::Precision::fp16) {
                operators_.project_tokens_fp16(
                    projected.buffer, encoded.features[index],
                    projection_weight.half_buffer, projection_bias,
                    encoded.patch_width, encoded.patch_height, embedding_,
                    project_channels_[index]);
            } else if (precision_ == inferbridge::native::Precision::int8) {
                operators_.project_tokens_int8(
                    projected.buffer, encoded.features[index],
                    projection_weight.int8_buffer,
                    projection_weight.int8_scales, projection_bias,
                    encoded.patch_width, encoded.patch_height, embedding_,
                    project_channels_[index]);
            } else {
                operators_.project_tokens(
                    projected.buffer, encoded.features[index],
                    projection_weight.buffer, projection_bias,
                    encoded.patch_width, encoded.patch_height, embedding_,
                    project_channels_[index], false);
            }
            if (index < 2) {
                const std::uint32_t kernel = index == 0 ? 4 : 2;
                FeatureMap resized{
                    context_.create_device_buffer(
                        elements(
                            projected.width * kernel,
                            projected.height * kernel,
                            projected.channels) *
                        sizeof(float)),
                    projected.width * kernel,
                    projected.height * kernel,
                    projected.channels,
                };
                const std::string resize =
                    "depth_head.resize_layers." + std::to_string(index);
                const GpuTensor& resize_weight =
                    weights_.tensor(resize + ".weight");
                const VulkanBuffer& resize_bias =
                    weight(weights_, resize + ".bias");
                if (precision_ == inferbridge::native::Precision::fp16) {
                    operators_.conv_transpose_nonoverlap_fp16(
                        resized.buffer, projected.buffer,
                        resize_weight.half_buffer, resize_bias,
                        projected.width, projected.height, projected.channels,
                        projected.channels, kernel);
                } else if (
                    precision_ == inferbridge::native::Precision::int8) {
                    operators_.conv_transpose_nonoverlap_int8(
                        resized.buffer, projected.buffer,
                        resize_weight.int8_buffer,
                        resize_weight.int8_scales, resize_bias,
                        projected.width, projected.height, projected.channels,
                        projected.channels, kernel);
                } else {
                    operators_.conv_transpose_nonoverlap(
                        resized.buffer, projected.buffer,
                        resize_weight.buffer, resize_bias,
                        projected.width, projected.height, projected.channels,
                        projected.channels, kernel, false);
                }
                layers[index] = std::move(resized);
            } else if (index == 2) {
                layers[index] = std::move(projected);
            } else {
                layers[index] = conv(
                    std::move(projected),
                    "depth_head.resize_layers.3.weight",
                    "depth_head.resize_layers.3.bias",
                    project_channels_[3],
                    3,
                    2,
                    1,
                    true);
            }
            });
        }
    };
    if (encoded.tokens > 2000) {
        run_projections();
    } else {
        context_.batch(run_projections);
    }

    FeatureMap refined[4];
    FeatureMap path;
    FeatureMap depth;
    const auto run_head = [&] {
    context_.batch([&] {
        for (std::uint32_t index = 0; index < 4; ++index) {
            const std::string prefix =
                "depth_head.scratch.layer" + std::to_string(index + 1) +
                "_rn.weight";
            refined[index] = conv(
                std::move(layers[index]),
                prefix,
                "",
                features_,
                3,
                1,
                1,
                false);
        }
    });

    context_.batch([&] {
        path = fusion(
            std::move(refined[3]),
            FeatureMap{},
            "depth_head.scratch.refinenet4",
            refined[2].width,
            refined[2].height);
    });
    context_.batch([&] {
        path = fusion(
            std::move(path),
            std::move(refined[2]),
            "depth_head.scratch.refinenet3",
            refined[1].width,
            refined[1].height);
    });
    context_.batch([&] {
        path = fusion(
            std::move(path),
            std::move(refined[1]),
            "depth_head.scratch.refinenet2",
            refined[0].width,
            refined[0].height);
    });
    context_.batch([&] {
        path = fusion(
            std::move(path),
            std::move(refined[0]),
            "depth_head.scratch.refinenet1",
            refined[0].width * 2,
            refined[0].height * 2);
    });

    context_.batch([&] {
        path = conv(
            std::move(path),
            "depth_head.scratch.output_conv1.weight",
            "depth_head.scratch.output_conv1.bias",
            features_ / 2,
            3,
            1,
            1,
            true);
        FeatureMap full{
            context_.create_device_buffer(
                elements(
                    encoded.patch_width * 14,
                    encoded.patch_height * 14,
                    features_ / 2) *
                sizeof(float)),
            encoded.patch_width * 14,
            encoded.patch_height * 14,
            features_ / 2,
        };
        operators_.bilinear_align_true(
            full.buffer,
            path.buffer,
            path.width,
            path.height,
            full.width,
            full.height,
            full.channels);
        full = conv(
            std::move(full),
            "depth_head.scratch.output_conv2.0.weight",
            "depth_head.scratch.output_conv2.0.bias",
            32,
            3,
            1,
            1,
            true);
        operators_.relu(
            full.buffer,
            full.buffer,
            static_cast<std::uint32_t>(
                elements(full.width, full.height, full.channels)));
        depth = conv(
            std::move(full),
            "depth_head.scratch.output_conv2.2.weight",
            "depth_head.scratch.output_conv2.2.bias",
            1,
            1,
            1,
            0,
            true);
        if (metric_max_depth_ > 0.0f) {
            operators_.sigmoid_scale(
                depth.buffer, depth.buffer,
                depth.width * depth.height,
                metric_max_depth_);
        } else {
            operators_.relu(
                depth.buffer,
                depth.buffer,
                depth.width * depth.height);
        }
    });
    };
    if (encoded.tokens > 2000) {
        run_head();
    } else {
        context_.batch(run_head);
    }
    return depth;
}

}  // namespace dav2
