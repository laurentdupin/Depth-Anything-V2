#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) readonly buffer Source { float data[]; } source_data;
layout(std430, binding = 1) writeonly buffer Range { float data[]; } range_data;
layout(push_constant) uniform Parameters { uint count; } parameters;
shared float minima[256];
shared float maxima[256];

void main() {
    uint lane = gl_LocalInvocationID.x;
    float minimum = 1.0 / 0.0;
    float maximum = -1.0 / 0.0;
    for (uint index = lane; index < parameters.count; index += 256u) {
        float value = source_data.data[index];
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }
    minima[lane] = minimum;
    maxima[lane] = maximum;
    barrier();
    for (uint step = 128u; step > 0u; step >>= 1u) {
        if (lane < step) {
            minima[lane] = min(minima[lane], minima[lane + step]);
            maxima[lane] = max(maxima[lane], maxima[lane + step]);
        }
        barrier();
    }
    if (lane == 0u) {
        range_data.data[0] = minima[0];
        range_data.data[1] = maxima[0];
    }
}
