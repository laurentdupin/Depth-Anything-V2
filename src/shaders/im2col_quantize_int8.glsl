#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 1, std430) writeonly buffer Packed {
    int data[];
} packed_buffer;
layout(set = 0, binding = 2, std430) writeonly buffer Scale {
    float data[];
} scale_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint input_channels;
    uint output_width;
    uint output_height;
    uint kernel;
    uint stride;
    int padding;
} parameters;
shared float maxima[256];
shared float row_scale;

float read_im2col(uint row, uint column) {
    const uint output_x = row % parameters.output_width;
    const uint output_y = row / parameters.output_width;
    const uint kernel_area = parameters.kernel * parameters.kernel;
    const uint input_channel = column / kernel_area;
    const uint kernel_index = column % kernel_area;
    const uint kernel_x = kernel_index % parameters.kernel;
    const uint kernel_y = kernel_index / parameters.kernel;
    const int input_x = int(output_x * parameters.stride + kernel_x) -
        parameters.padding;
    const int input_y = int(output_y * parameters.stride + kernel_y) -
        parameters.padding;
    if (input_x < 0 || input_y < 0 ||
        input_x >= int(parameters.input_width) ||
        input_y >= int(parameters.input_height)) return 0.0;
    return input_buffer.data[
        (input_channel * parameters.input_height + uint(input_y)) *
            parameters.input_width + uint(input_x)];
}

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint lane = gl_LocalInvocationID.x;
    const uint rows = parameters.output_width * parameters.output_height;
    const uint columns = parameters.input_channels *
        parameters.kernel * parameters.kernel;
    if (row >= rows) return;
    float maximum = 0.0;
    for (uint column = lane; column < columns; column += 256u)
        maximum = max(maximum, abs(read_im2col(row, column)));
    maxima[lane] = maximum;
    barrier();
    for (uint step = 128u; step > 0u; step >>= 1u) {
        if (lane < step) maxima[lane] = max(maxima[lane], maxima[lane + step]);
        barrier();
    }
    if (lane == 0u) {
        row_scale = max(maxima[0] / 127.0, 1.0e-8);
        scale_buffer.data[row] = row_scale;
    }
    barrier();
    const uint packed_columns = (columns + 3u) / 4u;
    const float inverse_scale = 1.0 / row_scale;
    for (uint packed_column = lane;
         packed_column < packed_columns;
         packed_column += 256u) {
        uint packed = 0u;
        for (uint component = 0u; component < 4u; ++component) {
            const uint column = packed_column * 4u + component;
            const int value = column < columns
                ? int(round(clamp(
                    read_im2col(row, column) * inverse_scale,
                    -127.0, 127.0)))
                : 0;
            packed |= (uint(value) & 0xffu) << (component * 8u);
        }
        packed_buffer.data[row * packed_columns + packed_column] = int(packed);
    }
}
