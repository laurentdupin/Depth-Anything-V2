#include "encoder.h"

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
    const GpuModel& weights,
    VulkanOperators& operators)
    : encoder_(encoder),
      context_(context),
      weights_(weights),
      operators_(operators) {
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
        operators_.prepare_tokens(
            current,
            image,
            buffer(weights_, "pretrained.patch_embed.proj.weight"),
            buffer(weights_, "pretrained.patch_embed.proj.bias"),
            buffer(weights_, "pretrained.cls_token"),
            buffer(weights_, "pretrained.pos_embed"),
            width,
            height,
            embedding_);
    });

    EncoderOutput result;
    result.features.reserve(4);
    result.patch_width = patch_width;
    result.patch_height = patch_height;
    result.tokens = tokens;
    result.embedding = embedding_;
    std::uint32_t capture_index = 0;

    for (std::uint32_t block = 0; block < blocks_; ++block) {
        context_.batch([&] {
            operators_.layer_norm(
                normalized,
                current,
                buffer(weights_, block_name(block, ".norm1.weight")),
                buffer(weights_, block_name(block, ".norm1.bias")),
                tokens,
                embedding_,
                1.0e-6f);
            operators_.linear(
                qkv,
                normalized,
                buffer(weights_, block_name(block, ".attn.qkv.weight")),
                buffer(weights_, block_name(block, ".attn.qkv.bias")),
                tokens,
                embedding_,
                embedding_ * 3,
                false);
            operators_.attention_head64(
                attention, qkv, tokens, heads_);
            operators_.linear(
                query,
                attention,
                buffer(weights_, block_name(block, ".attn.proj.weight")),
                buffer(weights_, block_name(block, ".attn.proj.bias")),
                tokens,
                embedding_,
                embedding_,
                false);
            operators_.add_scaled(
                next,
                current,
                query,
                buffer(weights_, block_name(block, ".ls1.gamma")),
                static_cast<std::uint32_t>(token_elements),
                embedding_);
            std::swap(current, next);

            operators_.layer_norm(
                normalized,
                current,
                buffer(weights_, block_name(block, ".norm2.weight")),
                buffer(weights_, block_name(block, ".norm2.bias")),
                tokens,
                embedding_,
                1.0e-6f);
            operators_.linear(
                hidden,
                normalized,
                buffer(weights_, block_name(block, ".mlp.fc1.weight")),
                buffer(weights_, block_name(block, ".mlp.fc1.bias")),
                tokens,
                embedding_,
                embedding_ * 4,
                true);
            operators_.linear(
                query,
                hidden,
                buffer(weights_, block_name(block, ".mlp.fc2.weight")),
                buffer(weights_, block_name(block, ".mlp.fc2.bias")),
                tokens,
                embedding_ * 4,
                embedding_,
                false);
            operators_.add_scaled(
                next,
                current,
                query,
                buffer(weights_, block_name(block, ".ls2.gamma")),
                static_cast<std::uint32_t>(token_elements),
                embedding_);
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
        });
    }
    if (result.features.size() != 4) {
        throw std::runtime_error("encoder did not produce four features");
    }
    return result;
}

}  // namespace dav2
