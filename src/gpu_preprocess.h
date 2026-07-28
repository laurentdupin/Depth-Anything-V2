#pragma once

#include "vulkan.h"

#include <cstdint>

namespace dav2 {

enum class GpuPixelOrder : std::uint32_t {
    bgra = 0,
    rgba = 1,
};

class GpuPreprocessor {
public:
    explicit GpuPreprocessor(VulkanContext& context);

    void run(
        VulkanBuffer& destination,
        const VulkanBuffer& source,
        std::uint32_t source_width,
        std::uint32_t source_height,
        std::uint32_t source_row_stride_bytes,
        std::uint32_t destination_width,
        std::uint32_t destination_height,
        GpuPixelOrder order,
        const VulkanSemaphore* wait = nullptr);

private:
    VulkanContext& context_;
    VulkanPipeline pipeline_;
};

}  // namespace dav2
