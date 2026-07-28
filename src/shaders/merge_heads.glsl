#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;

layout(push_constant) uniform Parameters {
    uint tokens;
    uint heads;
} parameters;

void main() {
    const uint feature = gl_GlobalInvocationID.x;
    const uint token = gl_GlobalInvocationID.y;
    const uint head = gl_GlobalInvocationID.z;
    if (feature >= 64 || token >= parameters.tokens ||
        head >= parameters.heads) {
        return;
    }
    output_buffer.data[
        token * parameters.heads * 64 + head * 64 + feature] =
        input_buffer.data[
            (head * parameters.tokens + token) * 64 + feature];
}
