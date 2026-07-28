#version 450 core

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;

layout(push_constant) uniform Parameters {
    uint rows;
    uint input_columns;
    uint output_columns;
} parameters;

shared float input_tile[16][16];
shared float weight_tile[16][16];

float gelu_tanh(float value) {
    const float cube = value * value * value;
    const float inner =
        0.7978845608028654 * (value + 0.044715 * cube);
    return 0.5 * value *
        (1.0 + tanh(clamp(inner, -15.0, 15.0)));
}

void main() {
    const uint output_column = gl_GlobalInvocationID.x;
    const uint output_row = gl_GlobalInvocationID.y;
    const uint local_column = gl_LocalInvocationID.x;
    const uint local_row = gl_LocalInvocationID.y;
    const bool valid_output =
        output_column < parameters.output_columns &&
        output_row < parameters.rows;
    float sum = 0.0;

    for (uint base = 0; base < parameters.input_columns; base += 16) {
        const uint input_column = base + local_column;
        input_tile[local_row][local_column] =
            output_row < parameters.rows &&
                    input_column < parameters.input_columns
                ? input_buffer.data[
                      output_row * parameters.input_columns + input_column]
                : 0.0;
        const uint weight_row = base + local_row;
        weight_tile[local_row][local_column] =
            weight_row < parameters.input_columns &&
                    output_column < parameters.output_columns
                ? weight_buffer.data[
                      output_column * parameters.input_columns + weight_row]
                : 0.0;
        barrier();
        if (valid_output) {
            for (uint index = 0; index < 16; ++index) {
                sum += input_tile[local_row][index] *
                    weight_tile[index][local_column];
            }
        }
        barrier();
    }
    if (valid_output) {
        const float value = sum + bias_buffer.data[output_column];
        output_buffer.data[
            output_row * parameters.output_columns + output_column] =
            gelu_tanh(value);
    }
}
