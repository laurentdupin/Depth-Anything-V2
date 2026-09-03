#include "gpu_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dav2 {
namespace {

bool use_half_weight(std::string_view name) {
    constexpr std::string_view suffix = ".weight";
    const bool weight =
        name.size() >= suffix.size() &&
        name.substr(name.size() - suffix.size()) == suffix;
    if (!weight) {
        return false;
    }
    if (name == "pretrained.patch_embed.proj.weight" ||
        name.rfind("depth_head.", 0) == 0) {
        return true;
    }
    if (name.rfind("pretrained.blocks.", 0) != 0) {
        return false;
    }
    return name.find(".attn.") != std::string_view::npos ||
        name.find(".mlp.") != std::string_view::npos;
}

bool use_int8_weight(std::string_view name, const TensorView& tensor) {
    if (!use_half_weight(name) || tensor.dimensions[0] == 0) return false;
    if (tensor.rank == 2) return tensor.dimensions[1] % 4 == 0;
    if (tensor.rank != 4) return false;
    if (name == "depth_head.resize_layers.0.weight" ||
        name == "depth_head.resize_layers.1.weight") {
        return tensor.dimensions[0] % 4 == 0;
    }
    return (tensor.elements / tensor.dimensions[0]) % 4 == 0;
}

bool transposed_convolution_weight(std::string_view name) {
    return name == "depth_head.resize_layers.0.weight" ||
        name == "depth_head.resize_layers.1.weight";
}

std::uint16_t float_to_half(float input) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &input, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return static_cast<std::uint16_t>(
            sign | (mantissa != 0 ? 0x7e00u : 0x7c00u));
    }
    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        const std::uint32_t shift =
            static_cast<std::uint32_t>(14 - half_exponent);
        std::uint32_t rounded = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (rounded & 1u) != 0)) {
            ++rounded;
        }
        return static_cast<std::uint16_t>(sign | rounded);
    }
    std::uint32_t rounded_mantissa = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u ||
        (remainder == 0x1000u && (rounded_mantissa & 1u) != 0)) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400u) {
            rounded_mantissa = 0;
            if (half_exponent + 1 >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            }
            return static_cast<std::uint16_t>(
                sign | ((half_exponent + 1) << 10));
        }
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(half_exponent) << 10) |
        rounded_mantissa);
}

}  // namespace

std::uint32_t crc32(const void* data, std::size_t bytes) {
    if (bytes != 0 && data == nullptr) {
        throw std::invalid_argument("CRC32 input is null");
    }
    auto* current = static_cast<const unsigned char*>(data);
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> result{};
        for (std::uint32_t byte = 0; byte < result.size(); ++byte) {
            std::uint32_t entry = byte;
            for (int bit = 0; bit < 8; ++bit) {
                const std::uint32_t mask =
                    0u - static_cast<std::uint32_t>(entry & 1u);
                entry = (entry >> 1) ^ (0xedb88320u & mask);
            }
            result[byte] = entry;
        }
        return result;
    }();
    std::uint32_t value = 0xffffffffu;
    for (std::size_t index = 0; index < bytes; ++index) {
        value = table[(value ^ current[index]) & 0xffu] ^ (value >> 8);
    }
    return value ^ 0xffffffffu;
}

GpuModel::GpuModel(
    const ModelFile& model,
    VulkanContext& context,
    inferbridge::native::Precision precision)
    : context_(context) {
    tensors_.reserve(model.tensor_count());
    for (std::string_view name : model.tensor_names()) {
        const TensorView& source = model.tensor(name);
        if (source.elements >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error(
                "model tensor is too large for this process: " +
                std::string(name));
        }
        const std::size_t bytes =
            static_cast<std::size_t>(source.elements) * sizeof(float);
        if (crc32(source.data, bytes) != source.crc32) {
            throw std::runtime_error(
                "model tensor checksum mismatch: " + std::string(name));
        }
        GpuTensor destination{
            context.create_device_buffer(bytes),
            {},
            {},
            {},
            {},
            {},
            source.dimensions,
            source.rank,
            source.elements,
        };
        context.upload(destination.buffer, source.data, bytes);
        if (use_half_weight(name) &&
            (precision != inferbridge::native::Precision::int8 ||
             name.rfind("depth_head.", 0) == 0)) {
            const auto* floats =
                static_cast<const float*>(source.data);
            std::vector<std::uint32_t> packed(
                static_cast<std::size_t>((source.elements + 1) / 2),
                0);
            for (std::uint64_t index = 0;
                 index < source.elements;
                 ++index) {
                packed[static_cast<std::size_t>(index / 2)] |=
                    static_cast<std::uint32_t>(
                        float_to_half(floats[index])) <<
                    ((index & 1u) * 16u);
            }
            destination.half_buffer = context.create_device_buffer(
                packed.size() * sizeof(std::uint32_t));
            context.upload(
                destination.half_buffer,
                packed.data(),
                packed.size() * sizeof(std::uint32_t));
            if (source.rank == 2 &&
                name.rfind("pretrained.blocks.", 0) == 0) {
                const std::size_t output_columns =
                    static_cast<std::size_t>(source.dimensions[0]);
                const std::size_t input_columns =
                    static_cast<std::size_t>(source.dimensions[1]);
                std::vector<std::uint16_t> transposed(
                    static_cast<std::size_t>(source.elements));
                for (std::size_t output = 0;
                     output < output_columns; ++output) {
                    for (std::size_t input = 0;
                         input < input_columns; ++input) {
                        transposed[input * output_columns + output] =
                            float_to_half(
                                floats[output * input_columns + input]);
                    }
                }
                destination.half_transposed_buffer =
                    context.create_device_buffer(
                        transposed.size() * sizeof(std::uint16_t));
                context.upload(
                    destination.half_transposed_buffer,
                    transposed.data(),
                    transposed.size() * sizeof(std::uint16_t));
            }
        }
        if (precision == inferbridge::native::Precision::int8 &&
            use_int8_weight(name, source)) {
            const auto* floats = source.data;
            const bool transposed = transposed_convolution_weight(name);
            const std::size_t input_columns = transposed
                ? static_cast<std::size_t>(source.dimensions[0])
                : static_cast<std::size_t>(
                    source.elements / source.dimensions[0]);
            const std::size_t output_rows = transposed
                ? static_cast<std::size_t>(source.dimensions[1]) *
                    source.dimensions[2] * source.dimensions[3]
                : static_cast<std::size_t>(source.dimensions[0]);
            std::vector<float> scales(output_rows, 1.0e-8f);
            std::vector<std::uint32_t> packed(
                output_rows * (input_columns / 4), 0u);
            const bool make_transposed = source.rank == 2 &&
                name.rfind("pretrained.blocks.", 0) == 0;
            std::vector<std::int8_t> transposed_int8(
                make_transposed ? source.elements : 0);
            const auto source_value = [&](std::size_t output,
                                          std::size_t input) {
                if (!transposed) {
                    return floats[output * input_columns + input];
                }
                const std::size_t output_channels = source.dimensions[1];
                const std::size_t kernel = source.dimensions[2];
                const std::size_t kernel_index = output / output_channels;
                const std::size_t output_channel = output % output_channels;
                const std::size_t kernel_y = kernel_index / kernel;
                const std::size_t kernel_x = kernel_index % kernel;
                return floats[
                    ((input * output_channels + output_channel) * kernel +
                        kernel_y) * kernel + kernel_x];
            };
            for (std::size_t output = 0; output < output_rows; ++output) {
                float maximum = 0.0f;
                for (std::size_t input = 0; input < input_columns; ++input) {
                    maximum = std::max(maximum, std::abs(source_value(output, input)));
                }
                const float scale = std::max(maximum / 127.0f, 1.0e-8f);
                scales[output] = scale;
                for (std::size_t input = 0; input < input_columns; input += 4) {
                    std::uint32_t word = 0u;
                    for (std::size_t lane = 0; lane < 4; ++lane) {
                        const float normalized =
                            source_value(output, input + lane) / scale;
                        const int quantized = static_cast<int>(std::nearbyint(
                            std::max(-127.0f, std::min(127.0f, normalized))));
                        word |= (static_cast<std::uint32_t>(quantized) & 0xffu)
                            << (lane * 8u);
                        if (make_transposed) {
                            transposed_int8[(input + lane) * output_rows + output] =
                                static_cast<std::int8_t>(quantized);
                        }
                    }
                    packed[output * (input_columns / 4) + input / 4] = word;
                }
            }
            destination.int8_buffer = context.create_device_buffer(
                packed.size() * sizeof(std::uint32_t));
            destination.int8_scales = context.create_device_buffer(
                scales.size() * sizeof(float));
            context.upload(
                destination.int8_buffer,
                packed.data(),
                packed.size() * sizeof(std::uint32_t));
            if (make_transposed) {
                destination.int8_transposed_buffer =
                    context.create_device_buffer(transposed_int8.size());
                context.upload(
                    destination.int8_transposed_buffer,
                    transposed_int8.data(), transposed_int8.size());
            }
            context.upload(
                destination.int8_scales,
                scales.data(),
                scales.size() * sizeof(float));
        }
        if (!tensors_.emplace(name, std::move(destination)).second) {
            throw std::runtime_error(
                "duplicate GPU tensor name: " + std::string(name));
        }
    }
}

const GpuTensor& GpuModel::tensor(std::string_view name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::runtime_error(
            "GPU model is missing tensor: " + std::string(name));
    }
    return found->second;
}

void GpuModel::retain_transformer_precision(
    inferbridge::native::Precision precision) {
    for (auto& entry : tensors_) {
        const std::string_view name = entry.first;
        const bool patch = name == "pretrained.patch_embed.proj.weight";
        if ((!patch && name.rfind("pretrained.blocks.", 0) != 0) ||
            name.size() < 7 ||
            name.substr(name.size() - 7) != ".weight" ||
            (!patch && name.find(".attn.") == std::string_view::npos &&
             name.find(".mlp.") == std::string_view::npos)) {
            continue;
        }
        GpuTensor& tensor = entry.second;
        if (precision != inferbridge::native::Precision::fp32) {
            context_.discard(tensor.buffer);
        }
        if (precision != inferbridge::native::Precision::fp16) {
            context_.discard(tensor.half_buffer);
            context_.discard(tensor.half_transposed_buffer);
        }
        if (precision != inferbridge::native::Precision::int8) {
            context_.discard(tensor.int8_buffer);
            context_.discard(tensor.int8_transposed_buffer);
            context_.discard(tensor.int8_scales);
        }
    }
}

void GpuModel::retain_dpt_precision(
    inferbridge::native::Precision projection_precision,
    inferbridge::native::Precision refinement_precision) {
    for (auto& entry : tensors_) {
        const std::string_view name = entry.first;
        if (name.rfind("depth_head.", 0) != 0 ||
            name.size() < 7 ||
            name.substr(name.size() - 7) != ".weight") {
            continue;
        }
        GpuTensor& tensor = entry.second;
        const inferbridge::native::Precision precision =
            name.rfind("depth_head.projects.", 0) == 0 ||
                name.rfind("depth_head.resize_layers.", 0) == 0
            ? projection_precision : refinement_precision;
        if (precision != inferbridge::native::Precision::fp32) {
            context_.discard(tensor.buffer);
        }
        if (precision != inferbridge::native::Precision::fp16) {
            context_.discard(tensor.half_buffer);
        }
        if (precision != inferbridge::native::Precision::int8) {
            context_.discard(tensor.int8_buffer);
            context_.discard(tensor.int8_transposed_buffer);
            context_.discard(tensor.int8_scales);
        }
    }
}

}  // namespace dav2
