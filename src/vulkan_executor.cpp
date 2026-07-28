#include "executor.h"
#include "gpu_model.h"
#include "model.h"
#include "operators.h"
#include "vulkan.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace dav2 {
namespace {

class VulkanExecutor final : public Executor {
public:
    VulkanExecutor(
        const std::string& model_path,
        dav2_encoder encoder,
        int vulkan_device_index)
        : model_(model_path, encoder),
          context_(static_cast<std::uint32_t>(vulkan_device_index)),
          weights_(model_, context_),
          operators_(context_) {}

    void infer(const float*, int, int, float*) override {
        throw std::runtime_error(
            "custom DAV2 inference is not implemented yet");
    }

private:
    ModelFile model_;
    VulkanContext context_;
    GpuModel weights_;
    VulkanOperators operators_;
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
