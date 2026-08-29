#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 1, std430) writeonly buffer Scale {
    float data[];
} scale_buffer;
layout(push_constant) uniform Parameters {
    uint rows;
    uint columns;
    uint input_offset;
} parameters;
shared float maxima[256];

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint lane = gl_LocalInvocationID.x;
    if (row >= parameters.rows) return;
    float maximum = 0.0;
    for (uint column = lane; column < parameters.columns; column += 256u)
        maximum = max(maximum, abs(input_buffer.data[
            parameters.input_offset + row * parameters.columns + column]));
    maxima[lane] = maximum;
    barrier();
    for (uint step = 128u; step > 0u; step >>= 1u) {
        if (lane < step)
            maxima[lane] = max(maxima[lane], maxima[lane + step]);
        barrier();
    }
    if (lane == 0u)
        scale_buffer.data[row] = max(maxima[0] / 127.0, 1.0e-8);
}
