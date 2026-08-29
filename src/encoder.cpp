#include "encoder.h"
#include "inferbridge/native_harness_precision.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace dav2 {
namespace {

const VulkanBuffer& buffer(
    const GpuModel& weights,
    const std::string& name) {
    return weights.tensor(name).buffer;
}

std::string block_name(std::uint32_t block, const char* suffix) {
    return "pretrained.blocks." + std::to_string(block) + suffix;
}

}  // namespace

DinoEncoder::DinoEncoder(
    dav2_encoder encoder,
    VulkanContext& context,
    GpuModel& weights,
    VulkanOperators& operators,
    inferbridge::native::Precision precision,
    bool force_fp32_attention)
    : encoder_(encoder),
      context_(context),
      weights_(weights),
      operators_(operators),
      precision_(precision),
      force_fp32_attention_(force_fp32_attention) {
    switch (encoder_) {
        case DAV2_ENCODER_VITS:
            embedding_ = 384;
            heads_ = 6;
            blocks_ = 12;
            capture_[0] = 2;
            capture_[1] = 5;
            capture_[2] = 8;
            capture_[3] = 11;
            break;
        case DAV2_ENCODER_VITB:
            embedding_ = 768;
            heads_ = 12;
            blocks_ = 12;
            capture_[0] = 2;
            capture_[1] = 5;
            capture_[2] = 8;
            capture_[3] = 11;
            break;
        case DAV2_ENCODER_VITL:
            embedding_ = 1024;
            heads_ = 16;
            blocks_ = 24;
            capture_[0] = 4;
            capture_[1] = 11;
            capture_[2] = 17;
            capture_[3] = 23;
            break;
        default:
            throw std::invalid_argument("unsupported DINOv2 encoder");
    }
    if (weights_.tensor("pretrained.cls_token").elements != embedding_ ||
        weights_.tensor("pretrained.pos_embed").elements !=
            std::uint64_t(1370) * embedding_) {
        throw std::runtime_error("encoder tensor dimensions do not match");
    }
}

void DinoEncoder::select_linear_tile(std::uint32_t rows) {
    const VkDeviceSize work_bytes =
        std::uint64_t(rows) * embedding_ * 4 * sizeof(float);
    VulkanBuffer left = context_.create_device_buffer(work_bytes);
    VulkanBuffer right = context_.create_device_buffer(work_bytes);
    const auto selected_weight = [&](
        const std::string& name,
        bool half_weight) -> const VulkanBuffer& {
        const GpuTensor& tensor = weights_.tensor(name);
        return half_weight ? tensor.half_buffer : tensor.buffer;
    };
    const auto run = [&](
        bool block16,
        bool half_weight,
        bool vectorized,
        std::uint32_t vector_tile) {
        const auto start = std::chrono::steady_clock::now();
        context_.batch([&] {
            operators_.linear(
                right,
                left,
                selected_weight(
                    block_name(0, ".attn.qkv.weight"), half_weight),
                buffer(weights_, block_name(0, ".attn.qkv.bias")),
                rows,
                embedding_,
                embedding_ * 3,
                false,
                block16,
                half_weight,
                vectorized,
                vector_tile);
            operators_.linear(
                left,
                right,
                selected_weight(
                    block_name(0, ".attn.proj.weight"), half_weight),
                buffer(weights_, block_name(0, ".attn.proj.bias")),
                rows,
                embedding_,
                embedding_,
                false,
                block16,
                half_weight,
                vectorized,
                vector_tile);
            operators_.linear(
                right,
                left,
                selected_weight(
                    block_name(0, ".mlp.fc1.weight"), half_weight),
                buffer(weights_, block_name(0, ".mlp.fc1.bias")),
                rows,
                embedding_,
                embedding_ * 4,
                true,
                block16,
                half_weight,
                vectorized,
                vector_tile);
            operators_.linear(
                left,
                right,
                selected_weight(
                    block_name(0, ".mlp.fc2.weight"), half_weight),
                buffer(weights_, block_name(0, ".mlp.fc2.bias")),
                rows,
                embedding_ * 4,
                embedding_,
                false,
                block16,
                half_weight,
                vectorized,
                vector_tile);
        });
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };
    struct Candidate {
        bool block16;
        bool half_weight;
        bool vectorized;
        std::uint32_t vector_tile;
        std::array<double, 3> samples{};
    };
    std::array<Candidate, 10> candidates{{
        {false, false, false, 0, {}},
        {true, false, false, 0, {}},
        {true, false, true, 4, {}},
        {true, false, true, 8, {}},
        {true, false, true, 16, {}},
        {false, true, false, 0, {}},
        {true, true, false, 0, {}},
        {true, true, true, 4, {}},
        {true, true, true, 8, {}},
        {true, true, true, 16, {}},
    }};
    for (Candidate& candidate : candidates) {
        run(
            candidate.block16,
            candidate.half_weight,
            candidate.vectorized,
            candidate.vector_tile);
    }
    for (std::size_t sample = 0;
         sample < candidates[0].samples.size();
         ++sample) {
        if ((sample & 1u) == 0) {
            for (Candidate& candidate : candidates) {
                candidate.samples[sample] =
                    run(
                        candidate.block16,
                        candidate.half_weight,
                        candidate.vectorized,
                        candidate.vector_tile);
            }
        } else {
            for (auto candidate = candidates.rbegin();
                 candidate != candidates.rend();
                 ++candidate) {
                candidate->samples[sample] =
                    run(
                        candidate->block16,
                        candidate->half_weight,
                        candidate->vectorized,
                        candidate->vector_tile);
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
    Candidate* best = best_fp32;
    if (best->vectorized && best->vector_tile == 16) {
        for (Candidate& candidate : candidates) {
            if (candidate.half_weight == best->half_weight &&
                candidate.vectorized && candidate.vector_tile == 8 &&
                best->samples[1] >= candidate.samples[1] * 0.95) {
                best = &candidate;
                break;
            }
        }
    }
    linear_block16_ = best->block16;
    linear_vectorized_ = best->vectorized;
    linear_vector_tile_ = best->vector_tile;
    linear_half_weight_ = best->half_weight;
    weights_.retain_transformer_precision(precision_);
    linear_tile_selected_ = true;
}

const VulkanBuffer& DinoEncoder::linear_weight(
    const std::string& name) const {
    const GpuTensor& tensor = weights_.tensor(name);
    if (precision_ == inferbridge::native::Precision::fp16) {
        return tensor.half_buffer;
    }
    if (precision_ == inferbridge::native::Precision::int8) {
        return tensor.int8_buffer;
    }
    return tensor.buffer;
}

void DinoEncoder::linear(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const std::string& weight_name,
    const std::string& bias_name,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool gelu) {
    const GpuTensor& weight = weights_.tensor(weight_name);
    const VulkanBuffer& bias = buffer(weights_, bias_name);
    if (precision_ == inferbridge::native::Precision::fp16) {
        operators_.linear_fp16(
            output, input, weight.half_buffer, bias,
            rows, input_columns, output_columns, gelu);
        return;
    }
    if (precision_ == inferbridge::native::Precision::int8) {
        operators_.linear_int8(
            output, input, weight.int8_buffer, weight.int8_scales, bias,
            rows, input_columns, output_columns, gelu);
        return;
    }
    operators_.linear(
        output, input, weight.buffer, bias,
        rows, input_columns, output_columns, gelu,
        linear_block16_, false, linear_vectorized_, linear_vector_tile_);
}

void DinoEncoder::prepare(
    std::uint32_t width,
    std::uint32_t height) {
    if (width == 0 || height == 0 || width % 14 != 0 ||
        height % 14 != 0) {
        throw std::invalid_argument(
            "encoder dimensions must be positive multiples of 14");
    }
    if (!linear_tile_selected_ &&
        precision_ == inferbridge::native::Precision::fp32) {
        select_linear_tile((width / 14) * (height / 14) + 1);
    } else if (!linear_tile_selected_) {
        weights_.retain_transformer_precision(precision_);
        linear_tile_selected_ = true;
    }
}

EncoderOutput DinoEncoder::forward(
    const VulkanBuffer& image,
    std::uint32_t width,
    std::uint32_t height) {
    if (width == 0 || height == 0 || width % 14 != 0 || height % 14 != 0) {
        throw std::invalid_argument(
            "encoder dimensions must be positive multiples of 14");
    }
    const std::uint32_t patch_width = width / 14;
    const std::uint32_t patch_height = height / 14;
    const std::uint32_t tokens =
        patch_width * patch_height + 1;
    prepare(width, height);
    const std::uint64_t token_elements =
        std::uint64_t(tokens) * embedding_;
    const VkDeviceSize token_bytes = token_elements * sizeof(float);

    VulkanBuffer current = context_.create_device_buffer(token_bytes);
    VulkanBuffer next = context_.create_device_buffer(token_bytes);
    VulkanBuffer normalized = context_.create_device_buffer(token_bytes);
    VulkanBuffer query = context_.create_device_buffer(token_bytes);
    VulkanBuffer attention = context_.create_device_buffer(token_bytes);
    VulkanBuffer qkv =
        context_.create_device_buffer(token_bytes * 3);
    VulkanBuffer hidden =
        context_.create_device_buffer(token_bytes * 4);
    context_.batch([&] {
        const GpuTensor& patch_weight =
            weights_.tensor("pretrained.patch_embed.proj.weight");
        operators_.prepare_tokens(
            current,
            image,
            patch_weight.buffer,
            patch_weight.half_buffer,
            patch_weight.int8_buffer,
            patch_weight.int8_scales,
            buffer(weights_, "pretrained.patch_embed.proj.bias"),
            buffer(weights_, "pretrained.cls_token"),
            buffer(weights_, "pretrained.pos_embed"),
            width,
            height,
            embedding_,
            precision_);
    });
    const inferbridge::native::Precision attention_precision =
        force_fp32_attention_
        ? inferbridge::native::Precision::fp32
        : precision_ == inferbridge::native::Precision::int8
            ? (context_.compute_capabilities().fp16
                ? inferbridge::native::Precision::fp16
                : inferbridge::native::Precision::fp32)
            : precision_;
    const VkDeviceSize attention_score_bytes =
        attention_precision == inferbridge::native::Precision::fp16
        ? std::uint64_t(heads_) * tokens *
            ((std::uint64_t(tokens) + 1) / 2) *
            sizeof(std::uint32_t)
        : std::uint64_t(heads_) * tokens * tokens * sizeof(float);
    VulkanBuffer attention_scores =
        context_.create_device_buffer(attention_score_bytes);

    EncoderOutput result;
    result.features.reserve(4);
    result.patch_width = patch_width;
    result.patch_height = patch_height;
    result.tokens = tokens;
    result.embedding = embedding_;
    std::uint32_t capture_index = 0;

#if defined(__ANDROID__)
    constexpr std::uint32_t blocks_per_submission = 1;
#else
    const std::uint32_t blocks_per_submission =
        tokens > 2000 ? 4 : blocks_;
#endif
    for (std::uint32_t block_begin = 0;
         block_begin < blocks_;
         block_begin += blocks_per_submission) {
        const std::uint32_t block_end =
            std::min(blocks_, block_begin + blocks_per_submission);
        context_.batch([&, block_begin, block_end] {
        for (std::uint32_t block = block_begin;
             block < block_end;
             ++block) {
            operators_.layer_norm(
                normalized,
                current,
                buffer(weights_, block_name(block, ".norm1.weight")),
                buffer(weights_, block_name(block, ".norm1.bias")),
                tokens,
                embedding_,
                1.0e-6f);
            linear(
                qkv, normalized,
                block_name(block, ".attn.qkv.weight"),
                block_name(block, ".attn.qkv.bias"),
                tokens, embedding_, embedding_ * 3, false);
            operators_.attention_head64(
                attention,
                qkv,
                tokens,
                heads_,
                &attention_scores,
                attention_precision);
            if (precision_ == inferbridge::native::Precision::fp32 &&
                linear_vectorized_ &&
                (linear_vector_tile_ == 8 ||
                 linear_vector_tile_ == 16)) {
                operators_.linear_residual_wide(
                    next,
                    attention,
                    linear_weight(block_name(block, ".attn.proj.weight")),
                    buffer(weights_, block_name(block, ".attn.proj.bias")),
                    current,
                    buffer(weights_, block_name(block, ".ls1.gamma")),
                    tokens,
                    embedding_,
                    embedding_,
                    linear_vector_tile_,
                    linear_half_weight_);
            } else {
                linear(
                    query, attention,
                    block_name(block, ".attn.proj.weight"),
                    block_name(block, ".attn.proj.bias"),
                    tokens, embedding_, embedding_, false);
                operators_.add_scaled(
                    next,
                    current,
                    query,
                    buffer(weights_, block_name(block, ".ls1.gamma")),
                    static_cast<std::uint32_t>(token_elements),
                    embedding_);
            }
            std::swap(current, next);

            operators_.layer_norm(
                normalized,
                current,
                buffer(weights_, block_name(block, ".norm2.weight")),
                buffer(weights_, block_name(block, ".norm2.bias")),
                tokens,
                embedding_,
                1.0e-6f);
            linear(
                hidden, normalized,
                block_name(block, ".mlp.fc1.weight"),
                block_name(block, ".mlp.fc1.bias"),
                tokens, embedding_, embedding_ * 4, true);
            if (precision_ == inferbridge::native::Precision::fp32 &&
                linear_vectorized_ &&
                (linear_vector_tile_ == 8 ||
                 linear_vector_tile_ == 16)) {
                operators_.linear_residual_wide(
                    next,
                    hidden,
                    linear_weight(block_name(block, ".mlp.fc2.weight")),
                    buffer(weights_, block_name(block, ".mlp.fc2.bias")),
                    current,
                    buffer(weights_, block_name(block, ".ls2.gamma")),
                    tokens,
                    embedding_ * 4,
                    embedding_,
                    linear_vector_tile_,
                    linear_half_weight_);
            } else {
                linear(
                    query, hidden,
                    block_name(block, ".mlp.fc2.weight"),
                    block_name(block, ".mlp.fc2.bias"),
                    tokens, embedding_ * 4, embedding_, false);
                operators_.add_scaled(
                    next,
                    current,
                    query,
                    buffer(weights_, block_name(block, ".ls2.gamma")),
                    static_cast<std::uint32_t>(token_elements),
                    embedding_);
            }
            std::swap(current, next);

            if (capture_index < 4 && block == capture_[capture_index]) {
                VulkanBuffer feature =
                    context_.create_device_buffer(token_bytes);
                operators_.layer_norm(
                    feature,
                    current,
                    buffer(weights_, "pretrained.norm.weight"),
                    buffer(weights_, "pretrained.norm.bias"),
                    tokens,
                    embedding_,
                    1.0e-6f);
                result.features.push_back(std::move(feature));
                ++capture_index;
            }
        }
        });
    }
    if (result.features.size() != 4) {
        throw std::runtime_error("encoder did not produce four features");
    }
    return result;
}

}  // namespace dav2
