#version 450 core
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer ClassToken {
    float data[];
} class_token_buffer;
layout(push_constant) uniform Parameters { uint embedding; } parameters;
void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index < parameters.embedding)
        output_buffer.data[index] = class_token_buffer.data[index];
}
