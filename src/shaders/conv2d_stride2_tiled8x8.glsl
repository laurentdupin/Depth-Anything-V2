#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;

layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint input_channels;
    uint output_width;
    uint output_height;
    uint output_channels;
    uint kernel;
    uint stride;
    int padding;
    uint has_bias;
} parameters;

shared float spatial_tile[8 * 17 * 17];
shared float kernel_tile[8 * 8 * 9];

void main() {
    const uint output_x = gl_GlobalInvocationID.x;
    const uint output_y = gl_GlobalInvocationID.y;
    const uint output_channel_base = gl_WorkGroupID.z * 8;
    const bool valid =
        output_x < parameters.output_width &&
        output_y < parameters.output_height &&
        output_channel_base < parameters.output_channels;
    float sums[8];
    for (uint output_offset = 0; output_offset < 8; ++output_offset) {
        sums[output_offset] = 0.0;
    }
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    const int input_origin_x =
        int(gl_WorkGroupID.x * 16) - parameters.padding;
    const int input_origin_y =
        int(gl_WorkGroupID.y * 16) - parameters.padding;
    for (uint input_channel_base = 0;
         input_channel_base < parameters.input_channels;
         input_channel_base += 8) {
        for (uint index = lane;
             index < 8 * 17 * 17;
             index += 64) {
            const uint channel_offset = index / (17 * 17);
            const uint spatial_index = index % (17 * 17);
            const uint input_channel =
                input_channel_base + channel_offset;
            const int input_x =
                input_origin_x + int(spatial_index % 17);
            const int input_y =
                input_origin_y + int(spatial_index / 17);
            spatial_tile[index] =
                input_channel < parameters.input_channels &&
                input_x >= 0 &&
                input_x < int(parameters.input_width) &&
                input_y >= 0 &&
                input_y < int(parameters.input_height)
                ? input_buffer.data[
                    (input_channel * parameters.input_height +
                        uint(input_y)) *
                        parameters.input_width +
                    uint(input_x)]
                : 0.0;
        }
        for (uint index = lane;
             index < 8 * 8 * 9;
             index += 64) {
            const uint input_offset = index / (8 * 9);
            const uint remainder = index % (8 * 9);
            const uint output_offset = remainder / 9;
            const uint kernel_index = remainder % 9;
            const uint input_channel =
                input_channel_base + input_offset;
            const uint output_channel =
                output_channel_base + output_offset;
            kernel_tile[index] =
                input_channel < parameters.input_channels &&
                output_channel < parameters.output_channels
                ? weight_buffer.data[
                    (output_channel * parameters.input_channels +
                        input_channel) *
                        9 +
                    kernel_index]
                : 0.0;
        }
        memoryBarrierShared();
        barrier();
        if (valid) {
            for (uint input_offset = 0;
                 input_offset < 8 &&
                 input_channel_base + input_offset <
                    parameters.input_channels;
                 ++input_offset) {
                for (uint kernel_y = 0; kernel_y < 3; ++kernel_y) {
                    for (uint kernel_x = 0; kernel_x < 3; ++kernel_x) {
                        float input_value = spatial_tile[
                            input_offset * 17 * 17 +
                            (gl_LocalInvocationID.y * 2 + kernel_y) * 17 +
                            gl_LocalInvocationID.x * 2 + kernel_x];
                        const uint kernel_index =
                            kernel_y * 3 + kernel_x;
                        for (uint output_offset = 0;
                             output_offset < 8;
                             ++output_offset) {
                            sums[output_offset] += input_value *
                                kernel_tile[
                                    input_offset * 8 * 9 +
                                    output_offset * 9 +
                                    kernel_index];
                        }
                    }
                }
            }
        }
        memoryBarrierShared();
        barrier();
    }
    if (!valid) {
        return;
    }
    for (uint output_offset = 0; output_offset < 8; ++output_offset) {
        const uint output_channel =
            output_channel_base + output_offset;
        if (output_channel < parameters.output_channels) {
            output_buffer.data[
                (output_channel * parameters.output_height + output_y) *
                    parameters.output_width +
                output_x] =
                sums[output_offset] +
                (parameters.has_bias != 0
                    ? bias_buffer.data[output_channel]
                    : 0.0);
        }
    }
}
