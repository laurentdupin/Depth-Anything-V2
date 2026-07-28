#include "gpu_model.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace dav2 {

std::uint32_t crc32(const void* data, std::size_t bytes) {
    if (bytes != 0 && data == nullptr) {
        throw std::invalid_argument("CRC32 input is null");
    }
    auto* current = static_cast<const unsigned char*>(data);
    std::uint32_t value = 0xffffffffu;
    for (std::size_t index = 0; index < bytes; ++index) {
        value ^= current[index];
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask =
                0u - static_cast<std::uint32_t>(value & 1u);
            value = (value >> 1) ^ (0xedb88320u & mask);
        }
    }
    return value ^ 0xffffffffu;
}

GpuModel::GpuModel(const ModelFile& model, VulkanContext& context) {
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
            source.dimensions,
            source.rank,
            source.elements,
        };
        context.upload(destination.buffer, source.data, bytes);
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

}  // namespace dav2
