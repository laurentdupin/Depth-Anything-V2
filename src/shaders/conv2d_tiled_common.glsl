layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
#if defined(HALF_WEIGHT)
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    uint data[];
} weight_buffer;
float read_weight(uint index) {
    const vec2 values =
        unpackHalf2x16(weight_buffer.data[index >> 1]);
    return (index & 1) == 0 ? values.x : values.y;
}
#else
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
float read_weight(uint index) {
    return weight_buffer.data[index];
}
#endif
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

shared float spatial_tile[10 * 10];
shared float kernel_tile[OUTPUT_CHANNEL_BLOCK * 3 * 3];

void main() {
    const uint output_x = gl_GlobalInvocationID.x;
    const uint output_y = gl_GlobalInvocationID.y;
    const uint output_channel_base =
        gl_GlobalInvocationID.z * OUTPUT_CHANNEL_BLOCK;
    const bool valid_output =
        output_x < parameters.output_width &&
        output_y < parameters.output_height &&
        output_channel_base < parameters.output_channels;
    float sums[OUTPUT_CHANNEL_BLOCK];
    for (uint output_offset = 0;
         output_offset < OUTPUT_CHANNEL_BLOCK;
         ++output_offset) {
        sums[output_offset] = 0.0;
    }
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    const int input_origin_x = int(gl_WorkGroupID.x * 8) - 1;
    const int input_origin_y = int(gl_WorkGroupID.y * 8) - 1;
    for (uint input_channel = 0;
         input_channel < parameters.input_channels;
         ++input_channel) {
        for (uint index = lane; index < 100; index += 64) {
            const int input_x = input_origin_x + int(index % 10);
            const int input_y = input_origin_y + int(index / 10);
            spatial_tile[index] =
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
             index < OUTPUT_CHANNEL_BLOCK * 9;
             index += 64) {
            const uint output_offset = index / 9;
            const uint output_channel =
                output_channel_base + output_offset;
            kernel_tile[index] =
                output_channel < parameters.output_channels
                ? read_weight(
                    (output_channel * parameters.input_channels +
                        input_channel) *
                        9 +
                    index % 9)
                : 0.0;
        }
        barrier();
        if (valid_output) {
            const uint tile_x = gl_LocalInvocationID.x;
            const uint tile_y = gl_LocalInvocationID.y;
            for (uint kernel_y = 0; kernel_y < 3; ++kernel_y) {
                for (uint kernel_x = 0; kernel_x < 3; ++kernel_x) {
                    const float input_value = spatial_tile[
                        (tile_y + kernel_y) * 10 +
                        tile_x + kernel_x];
                    const uint kernel_index = kernel_y * 3 + kernel_x;
                    for (uint output_offset = 0;
                         output_offset < OUTPUT_CHANNEL_BLOCK;
                         ++output_offset) {
                        sums[output_offset] +=
                            input_value *
                            kernel_tile[output_offset * 9 + kernel_index];
                    }
                }
            }
        }
        barrier();
    }
    if (!valid_output) return;
    for (uint output_offset = 0;
         output_offset < OUTPUT_CHANNEL_BLOCK;
         ++output_offset) {
        const uint output_channel =
            output_channel_base + output_offset;
        if (output_channel < parameters.output_channels) {
            float sum = sums[output_offset];
            if (parameters.has_bias != 0) {
                sum += bias_buffer.data[output_channel];
            }
            output_buffer.data[
                (output_channel * parameters.output_height + output_y) *
                    parameters.output_width +
                output_x] = sum;
        }
    }
}
