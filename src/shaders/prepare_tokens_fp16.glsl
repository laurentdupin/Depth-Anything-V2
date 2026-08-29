#version 450 core
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer TokenOutput {
    float data[];
} token_output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Image {
    float data[];
} image_buffer;
layout(set = 0, binding = 2, std430) readonly buffer PatchWeight {
    uint data[];
} patch_weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer PatchBias {
    float data[];
} patch_bias_buffer;
layout(set = 0, binding = 4, std430) readonly buffer ClassToken {
    float data[];
} class_token_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint patch_width;
    uint patch_height;
    uint embedding;
} parameters;

float16_t read_weight(uint index) {
    const vec2 pair = unpackHalf2x16(patch_weight_buffer.data[index >> 1]);
    return float16_t((index & 1u) == 0u ? pair.x : pair.y);
}

void main() {
    const uint feature = gl_GlobalInvocationID.x;
    const uint token = gl_GlobalInvocationID.y;
    const uint patch_count = parameters.patch_width * parameters.patch_height;
    if (feature >= parameters.embedding || token > patch_count) return;
    if (token == 0u) {
        token_output_buffer.data[feature] = class_token_buffer.data[feature];
        return;
    }
    const uint patch_index = token - 1u;
    const uint patch_x = patch_index % parameters.patch_width;
    const uint patch_y = patch_index / parameters.patch_width;
    float sum = 0.0;
    for (uint channel = 0u; channel < 3u; ++channel) {
        for (uint kernel_y = 0u; kernel_y < 14u; ++kernel_y) {
            for (uint kernel_x = 0u; kernel_x < 14u; ++kernel_x) {
                const uint image_x = patch_x * 14u + kernel_x;
                const uint image_y = patch_y * 14u + kernel_y;
                const uint image_index =
                    (channel * parameters.input_height + image_y) *
                        parameters.input_width + image_x;
                const uint weight_index =
                    ((feature * 3u + channel) * 14u + kernel_y) * 14u +
                    kernel_x;
                sum += float(
                    float16_t(image_buffer.data[image_index]) *
                    read_weight(weight_index));
            }
        }
    }
    token_output_buffer.data[token * parameters.embedding + feature] =
        sum + patch_bias_buffer.data[feature];
}
