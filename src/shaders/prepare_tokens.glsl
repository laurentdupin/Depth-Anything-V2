#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Image {
    float data[];
} image_buffer;
layout(set = 0, binding = 2, std430) readonly buffer PatchWeight {
    float data[];
} patch_weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer PatchBias {
    float data[];
} patch_bias_buffer;
layout(set = 0, binding = 4, std430) readonly buffer ClassToken {
    float data[];
} class_token_buffer;
layout(set = 0, binding = 5, std430) readonly buffer Position {
    float data[];
} position_buffer;

layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint patch_width;
    uint patch_height;
    uint embedding;
} parameters;

float cubic_convolution1(float x) {
    const float a = -0.75;
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
}

float cubic_convolution2(float x) {
    const float a = -0.75;
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
}

vec4 cubic_coefficients(float fraction) {
    const float other = 1.0 - fraction;
    return vec4(
        cubic_convolution2(fraction + 1.0),
        cubic_convolution1(fraction),
        cubic_convolution1(other),
        cubic_convolution2(other + 1.0));
}

float position_value(int x, int y, uint feature) {
    const uint bounded_x = uint(clamp(x, 0, 36));
    const uint bounded_y = uint(clamp(y, 0, 36));
    return position_buffer.data[
        (1 + bounded_y * 37 + bounded_x) * parameters.embedding + feature];
}

float interpolated_position(uint x, uint y, uint feature) {
    if (parameters.patch_width == 37 &&
        parameters.patch_height == 37) {
        return position_value(int(x), int(y), feature);
    }
    const vec2 source =
        (vec2(x, y) + vec2(0.5)) *
            vec2(
                37.0 / (float(parameters.patch_width) + 0.1),
                37.0 / (float(parameters.patch_height) + 0.1)) -
        vec2(0.5);
    const ivec2 base = ivec2(floor(source));
    const vec2 fraction = source - vec2(base);
    const vec4 x_coefficients = cubic_coefficients(fraction.x);
    float rows[4];
    for (int row = 0; row < 4; ++row) {
        const int source_y = base.y - 1 + row;
        rows[row] =
            position_value(base.x - 1, source_y, feature) *
                x_coefficients.x +
            position_value(base.x, source_y, feature) *
                x_coefficients.y +
            position_value(base.x + 1, source_y, feature) *
                x_coefficients.z +
            position_value(base.x + 2, source_y, feature) *
                x_coefficients.w;
    }
    const vec4 y_coefficients = cubic_coefficients(fraction.y);
    return rows[0] * y_coefficients.x +
        rows[1] * y_coefficients.y +
        rows[2] * y_coefficients.z +
        rows[3] * y_coefficients.w;
}

void main() {
    const uint feature = gl_GlobalInvocationID.x;
    const uint token = gl_GlobalInvocationID.y;
    const uint patch_count =
        parameters.patch_width * parameters.patch_height;
    if (feature >= parameters.embedding || token > patch_count) {
        return;
    }
    if (token == 0) {
        output_buffer.data[feature] =
            class_token_buffer.data[feature] +
            position_buffer.data[feature];
        return;
    }

    const uint patch_index = token - 1;
    const uint patch_x = patch_index % parameters.patch_width;
    const uint patch_y = patch_index / parameters.patch_width;
    float sum = 0.0;
    for (uint channel = 0; channel < 3; ++channel) {
        for (uint kernel_y = 0; kernel_y < 14; ++kernel_y) {
            for (uint kernel_x = 0; kernel_x < 14; ++kernel_x) {
                const uint image_x = patch_x * 14 + kernel_x;
                const uint image_y = patch_y * 14 + kernel_y;
                const uint image_index =
                    (channel * parameters.input_height + image_y) *
                        parameters.input_width +
                    image_x;
                const uint weight_index =
                    ((feature * 3 + channel) * 14 + kernel_y) * 14 +
                    kernel_x;
                sum += image_buffer.data[image_index] *
                    patch_weight_buffer.data[weight_index];
            }
        }
    }
    output_buffer.data[token * parameters.embedding + feature] =
        sum + patch_bias_buffer.data[feature] +
        interpolated_position(patch_x, patch_y, feature);
}
