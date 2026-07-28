#include "elementwise_spv.h"
#include "vulkan.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    dav2::VulkanContext context(0);
    const std::vector<float> input{
        -4.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 7.0f, 12.0f,
    };
    auto gpu_input = context.create_device_buffer(input.size() * sizeof(float));
    auto gpu_output = context.create_device_buffer(input.size() * sizeof(float));
    context.upload(gpu_input, input.data(), input.size() * sizeof(float));

    auto pipeline = context.create_pipeline(
        dav2_elementwise_spv,
        dav2_elementwise_spv_size,
        2,
        16);
    struct Parameters {
        std::uint32_t count;
        float scale;
        float bias;
        std::uint32_t relu;
    } parameters{
        static_cast<std::uint32_t>(input.size()),
        1.5f,
        -0.25f,
        1,
    };
    context.dispatch(
        pipeline,
        {&gpu_input, &gpu_output},
        &parameters,
        sizeof(parameters),
        1);

    std::vector<float> output(input.size());
    context.download(
        gpu_output, output.data(), output.size() * sizeof(float));
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float expected =
            std::max(input[index] * parameters.scale + parameters.bias, 0.0f);
        assert(std::abs(output[index] - expected) == 0.0f);
    }
    std::cout << context.device_name() << '\n';
    return 0;
}
