#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Packed {
    int data[];
} packed_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) writeonly buffer Scale {
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
    for (uint column = lane; column < parameters.columns; column += 256u) {
        maximum = max(maximum, abs(input_buffer.data[
            parameters.input_offset + row * parameters.columns + column]));
    }
    maxima[lane] = maximum;
    barrier();
    for (uint step = 128u; step > 0u; step >>= 1u) {
        if (lane < step) {
            maxima[lane] = max(maxima[lane], maxima[lane + step]);
        }
        barrier();
    }
    const float scale = max(maxima[0] / 127.0, 1.0e-8);
    if (lane == 0u) scale_buffer.data[row] = scale;
    const uint packed_columns = parameters.columns / 4u;
    for (uint packed_column = lane;
         packed_column < packed_columns;
         packed_column += 256u) {
        const uint column_base = packed_column * 4u;
        uint packed = 0u;
        for (uint component = 0u; component < 4u; ++component) {
            const int value = int(round(clamp(
                input_buffer.data[
                    parameters.input_offset + row * parameters.columns +
                    column_base + component] / scale,
                -127.0, 127.0)));
            packed |= (uint(value) & 0xffu) << (component * 8u);
        }
        packed_buffer.data[row * packed_columns + packed_column] = int(packed);
    }
}
