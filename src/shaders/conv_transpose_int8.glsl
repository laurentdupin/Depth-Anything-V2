#version 460
#extension GL_EXT_integer_dot_product : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    int data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    int data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer InputScale {
    float data[];
} input_scale_buffer;
layout(set = 0, binding = 4, std430) readonly buffer WeightScale {
    float data[];
} weight_scale_buffer;
layout(set = 0, binding = 5, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint input_channels;
    uint output_channels;
    uint kernel;
} parameters;

void main() {
    const uint output_x = gl_GlobalInvocationID.x;
    const uint output_y = gl_GlobalInvocationID.y;
    const uint output_channel = gl_GlobalInvocationID.z;
    const uint output_width = parameters.input_width * parameters.kernel;
    const uint output_height = parameters.input_height * parameters.kernel;
    if (output_x >= output_width || output_y >= output_height ||
        output_channel >= parameters.output_channels) return;
    const uint input_x = output_x / parameters.kernel;
    const uint input_y = output_y / parameters.kernel;
    const uint kernel_x = output_x % parameters.kernel;
    const uint kernel_y = output_y % parameters.kernel;
    const uint input_row = input_y * parameters.input_width + input_x;
    const uint packed_columns = parameters.input_channels / 4u;
    const uint weight_row =
        (kernel_y * parameters.kernel + kernel_x) *
            parameters.output_channels + output_channel;
    int sum = 0;
    for (uint column = 0u; column < packed_columns; ++column) {
        sum += dotPacked4x8EXT(
            input_buffer.data[input_row * packed_columns + column],
            weight_buffer.data[weight_row * packed_columns + column]);
    }
    output_buffer.data[
        (output_channel * output_height + output_y) * output_width +
        output_x] = float(sum) * input_scale_buffer.data[input_row] *
        weight_scale_buffer.data[weight_row] + bias_buffer.data[output_channel];
}
