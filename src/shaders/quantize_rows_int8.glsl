#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Packed {
    int data[];
} packed_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Scale {
    float data[];
} scale_buffer;
layout(push_constant) uniform Parameters {
    uint rows;
    uint columns;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint packed_columns = parameters.columns / 4u;
    if (index >= parameters.rows * packed_columns) return;
    const uint row = index / packed_columns;
    const uint column_base = (index % packed_columns) * 4u;
    const float inverse_scale = 1.0 / scale_buffer.data[row];
    uint packed = 0u;
    for (uint lane = 0u; lane < 4u; ++lane) {
        const int value = int(round(clamp(
            input_buffer.data[row * parameters.columns + column_base + lane] *
                inverse_scale,
            -127.0, 127.0)));
        packed |= (uint(value) & 0xffu) << (lane * 8u);
    }
    packed_buffer.data[index] = int(packed);
}
