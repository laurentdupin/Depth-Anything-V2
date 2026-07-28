#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Query {
    float data[];
} query_buffer;
layout(set = 0, binding = 1, std430) writeonly buffer Key {
    float data[];
} key_buffer;
layout(set = 0, binding = 2, std430) writeonly buffer Value {
    float data[];
} value_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Qkv {
    float data[];
} qkv_buffer;

layout(push_constant) uniform Parameters {
    uint tokens;
    uint heads;
    uint embedding;
    float query_scale;
} parameters;

void main() {
    const uint feature = gl_GlobalInvocationID.x;
    const uint token = gl_GlobalInvocationID.y;
    const uint head = gl_GlobalInvocationID.z;
    if (feature >= 64 || token >= parameters.tokens ||
        head >= parameters.heads) {
        return;
    }
    const uint embedding_column = head * 64 + feature;
    const uint input_base = token * parameters.embedding * 3;
    const uint output_index =
        (head * parameters.tokens + token) * 64 + feature;
    query_buffer.data[output_index] =
        qkv_buffer.data[input_base + embedding_column] *
        parameters.query_scale;
    key_buffer.data[output_index] =
        qkv_buffer.data[input_base + parameters.embedding + embedding_column];
    value_buffer.data[output_index] =
        qkv_buffer.data[
            input_base + parameters.embedding * 2 + embedding_column];
}
