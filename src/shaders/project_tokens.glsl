#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Tokens {
    float data[];
} token_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;

layout(push_constant) uniform Parameters {
    uint width;
    uint height;
    uint embedding;
    uint output_channels;
} parameters;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint output_channel = gl_GlobalInvocationID.z;
    if (x >= parameters.width || y >= parameters.height ||
        output_channel >= parameters.output_channels) {
        return;
    }
    const uint token =
        1 + y * parameters.width + x;
    float sum = 0.0;
    for (uint input_channel = 0;
         input_channel < parameters.embedding;
         ++input_channel) {
        sum += token_buffer.data[
                   token * parameters.embedding + input_channel] *
            weight_buffer.data[
                output_channel * parameters.embedding + input_channel];
    }
    output_buffer.data[
        (output_channel * parameters.height + y) *
            parameters.width +
        x] = sum + bias_buffer.data[output_channel];
}
