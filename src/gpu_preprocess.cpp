#include "gpu_preprocess.h"

#include "preprocess_rgba_spv.h"

#include <limits>
#include <stdexcept>

namespace dav2 {

GpuPreprocessor::GpuPreprocessor(VulkanContext& context)
    : context_(context),
      pipeline_(context.create_pipeline(
          dav2_preprocess_rgba_spv,
          dav2_preprocess_rgba_spv_size,
          2,
          6 * sizeof(std::uint32_t))) {}

void GpuPreprocessor::run(
    VulkanBuffer& destination,
    const VulkanBuffer& source,
    std::uint32_t source_width,
    std::uint32_t source_height,
    std::uint32_t source_row_stride_bytes,
    std::uint32_t destination_width,
    std::uint32_t destination_height,
    GpuPixelOrder order,
    const VulkanSemaphore* wait) {
    if (source_width == 0 || source_height == 0 ||
        destination_width == 0 || destination_height == 0 ||
        source_row_stride_bytes % sizeof(std::uint32_t) != 0 ||
        source_row_stride_bytes <
            source_width * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            "invalid GPU preprocessing dimensions");
    }
    const std::uint64_t source_bytes =
        static_cast<std::uint64_t>(source_row_stride_bytes) *
        source_height;
    const std::uint64_t destination_bytes =
        static_cast<std::uint64_t>(destination_width) *
        destination_height * 3 * sizeof(float);
    if (source_bytes > source.size() ||
        destination_bytes > destination.size()) {
        throw std::invalid_argument(
            "GPU preprocessing buffer is too small");
    }
    if (source_width >
            static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
        source_height >
            static_cast<std::uint32_t>(
                std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "GPU preprocessing dimensions are too large");
    }
    struct Parameters {
        std::uint32_t source_width;
        std::uint32_t source_height;
        std::uint32_t source_row_stride_pixels;
        std::uint32_t destination_width;
        std::uint32_t destination_height;
        std::uint32_t rgba_order;
    } parameters{
        source_width,
        source_height,
        source_row_stride_bytes / sizeof(std::uint32_t),
        destination_width,
        destination_height,
        static_cast<std::uint32_t>(order),
    };
    context_.dispatch(
        pipeline_,
        {&source, &destination},
        &parameters,
        sizeof(parameters),
        (destination_width + 7) / 8,
        (destination_height + 7) / 8,
        1,
        wait);
}

}  // namespace dav2
