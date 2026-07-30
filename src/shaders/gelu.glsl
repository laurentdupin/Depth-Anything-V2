#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;

layout(push_constant) uniform Parameters {
    uint count;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.count) {
        return;
    }
    const float value = input_buffer.data[index];
    /*
     * torch.nn.GELU defaults to the exact erf formulation. GLSL 450 has no
     * erf intrinsic, so use the Numerical Recipes erfc approximation. Its
     * maximum scalar error is approximately 1.2e-7, unlike the former tanh
     * GELU approximation whose layer-by-layer drift was visible after the
     * worker's depth min/max normalization.
     */
    const float x = value * 0.7071067811865475;
    const float magnitude = abs(x);
    const float t = 1.0 / (1.0 + 0.5 * magnitude);
    const float polynomial =
        -1.26551223 +
        t * (1.00002368 +
        t * (0.37409196 +
        t * (0.09678418 +
        t * (-0.18628806 +
        t * (0.27886807 +
        t * (-1.13520398 +
        t * (1.48851587 +
        t * (-0.82215223 +
        t * 0.17087277))))))));
    const float erfc_magnitude =
        t * exp(-magnitude * magnitude + polynomial);
    const float erf_value =
        x >= 0.0 ? 1.0 - erfc_magnitude : erfc_magnitude - 1.0;
    output_buffer.data[index] =
        0.5 * value * (1.0 + erf_value);
}
