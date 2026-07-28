#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

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

void main() {
    const uint column_base = gl_GlobalInvocationID.x * 4;
    const uint row_base = gl_GlobalInvocationID.y * 4;
    if (column_base >= parameters.output_columns ||
        row_base >= parameters.rows) {
        return;
    }
    float sums[4][4];
    for (uint row = 0; row < 4; ++row) {
        for (uint column = 0; column < 4; ++column) {
            sums[row][column] = 0.0;
        }
    }
    for (uint inner = 0; inner < parameters.input_columns; ++inner) {
        float input_values[4];
        float weight_values[4];
        for (uint row = 0; row < 4; ++row) {
            const uint output_row = row_base + row;
            input_values[row] = output_row < parameters.rows
                ? input_buffer.data[
                      output_row * parameters.input_columns + inner]
                : 0.0;
        }
        for (uint column = 0; column < 4; ++column) {
            const uint output_column = column_base + column;
            weight_values[column] =
                output_column < parameters.output_columns
                ? weight_buffer.data[
                      output_column * parameters.input_columns + inner]
                : 0.0;
        }
        for (uint row = 0; row < 4; ++row) {
            for (uint column = 0; column < 4; ++column) {
                sums[row][column] +=
                    input_values[row] * weight_values[column];
            }
        }
    }
    for (uint row = 0; row < 4; ++row) {
        const uint output_row = row_base + row;
        if (output_row >= parameters.rows) {
            continue;
        }
        for (uint column = 0; column < 4; ++column) {
            const uint output_column = column_base + column;
            if (output_column < parameters.output_columns) {
                output_buffer.data[
                    output_row * parameters.output_columns + output_column] =
                    sums[row][column] + bias_buffer.data[output_column];
            }
        }
    }
}
