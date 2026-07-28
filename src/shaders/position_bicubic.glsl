#version 450 core

layout(std430) buffer;

layout(set = 0, binding = 0) restrict writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1) restrict readonly buffer Position {
    float data[];
} position_buffer;

layout(push_constant) uniform Parameters {
    uint patch_width;
    uint patch_height;
    uint embedding;
    float scale_x;
    float scale_y;
} parameters;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

float cubic_convolution1(float x, float a) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
}

float cubic_convolution2(float x, float a) {
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
}

vec4 cubic_coefficients(float t) {
    const float a = -0.75;
    const float x1 = t;
    const float x2 = 1.0 - t;
    return vec4(
        cubic_convolution2(x1 + 1.0, a),
        cubic_convolution1(x1, a),
        cubic_convolution1(x2, a),
        cubic_convolution2(x2 + 1.0, a));
}

float cubic_interpolate(
    float x0,
    float x1,
    float x2,
    float x3,
    float t) {
    const vec4 coefficients = cubic_coefficients(t);
    return x0 * coefficients.x + x1 * coefficients.y +
        x2 * coefficients.z + x3 * coefficients.w;
}

float fetch_bounded(ivec2 position, uint channel) {
    const ivec2 bounded =
        clamp(position, ivec2(0, 0), ivec2(36, 36));
    return position_buffer.data[
        (1 + uint(bounded.y) * 37 + uint(bounded.x)) *
            parameters.embedding +
        channel];
}

void main() {
    const uint write_index = gl_GlobalInvocationID.x;
    const uint output_elements =
        parameters.patch_width * parameters.patch_height *
        parameters.embedding;
    if (write_index >= output_elements) {
        return;
    }
    const uint output_x = write_index % parameters.patch_width;
    const uint output_y =
        (write_index / parameters.patch_width) %
        parameters.patch_height;
    const uint channel =
        write_index /
        (parameters.patch_width * parameters.patch_height);
    if (parameters.patch_width == 37 &&
        parameters.patch_height == 37) {
        output_buffer.data[write_index] =
            fetch_bounded(
                ivec2(int(output_x), int(output_y)), channel);
        return;
    }
    const vec2 source =
        (vec2(output_x, output_y) + vec2(0.5, 0.5)) *
            vec2(parameters.scale_x, parameters.scale_y) -
        vec2(0.5, 0.5);
    const ivec2 base = ivec2(floor(source));
    const vec2 fraction = source - vec2(base);
    float rows[4];
    for (int row = 0; row < 4; ++row) {
        const int y = base.y - 1 + row;
        rows[row] = cubic_interpolate(
            fetch_bounded(ivec2(base.x - 1, y), channel),
            fetch_bounded(ivec2(base.x, y), channel),
            fetch_bounded(ivec2(base.x + 1, y), channel),
            fetch_bounded(ivec2(base.x + 2, y), channel),
            fraction.x);
    }
    output_buffer.data[write_index] = cubic_interpolate(
        rows[0], rows[1], rows[2], rows[3], fraction.y);
}
