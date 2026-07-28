#include "executor.h"
#include "vulkan.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace dav2 {
namespace {

class VulkanExecutor final : public Executor {
public:
    VulkanExecutor(
        const std::string&,
        dav2_encoder,
        int vulkan_device_index)
        : context_(static_cast<std::uint32_t>(vulkan_device_index)) {
        throw std::runtime_error(
            "custom DAV2 model loading is not implemented yet");
    }

    void infer(const float*, int, int, float*) override {
        throw std::runtime_error(
            "custom DAV2 inference is not implemented yet");
    }

private:
    VulkanContext context_;
};

}  // namespace

std::unique_ptr<Executor> create_executor(
    const std::string& model_path,
    dav2_encoder encoder,
    int vulkan_device_index) {
    if (vulkan_device_index < 0) {
        throw std::invalid_argument("vulkan_device_index must be non-negative");
    }
    return std::make_unique<VulkanExecutor>(
        model_path, encoder, vulkan_device_index);
}

}  // namespace dav2
