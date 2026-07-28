#pragma once

#include "depth_anything_v2.h"

#include <memory>
#include <string>

namespace dav2 {

class Executor {
public:
    virtual ~Executor() = default;
    virtual void infer(
        const float* normalized_rgb_chw,
        int width,
        int height,
        float* depth) = 0;
};

std::unique_ptr<Executor> create_executor(
    const std::string& model_path,
    dav2_encoder encoder,
    int vulkan_device_index);

}  // namespace dav2
