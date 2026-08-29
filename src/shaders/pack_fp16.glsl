#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float16_t data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(push_constant) uniform Parameters { uint count; } parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index < parameters.count)
        output_buffer.data[index] = float16_t(input_buffer.data[index]);
}
