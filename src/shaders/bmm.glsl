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

layout(push_constant) uniform Parameters {
    uint rows;
    uint columns;
    uint inner;
    uint batches;
    uint weight_transposed;
} parameters;

void main() {
    const uint column_base = gl_GlobalInvocationID.x * 4;
    const uint row_base = gl_GlobalInvocationID.y * 8;
    const uint batch = gl_GlobalInvocationID.z;
    if (column_base >= parameters.columns ||
        row_base >= parameters.rows ||
        batch >= parameters.batches) {
        return;
    }
    float sums[8][4];
    for (uint row = 0; row < 8; ++row) {
        for (uint column = 0; column < 4; ++column) {
            sums[row][column] = 0.0;
        }
    }
    for (uint inner = 0; inner < parameters.inner; ++inner) {
        float input_values[8];
        float weight_values[4];
        for (uint row = 0; row < 8; ++row) {
            const uint output_row = row_base + row;
            input_values[row] = output_row < parameters.rows
                ? input_buffer.data[
                      (batch * parameters.rows + output_row) *
                          parameters.inner +
                      inner]
                : 0.0;
        }
        for (uint column = 0; column < 4; ++column) {
            const uint output_column = column_base + column;
            if (output_column < parameters.columns) {
                weight_values[column] = parameters.weight_transposed != 0
                    ? weight_buffer.data[
                          (batch * parameters.columns + output_column) *
                              parameters.inner +
                          inner]
                    : weight_buffer.data[
                          (batch * parameters.inner + inner) *
                              parameters.columns +
                          output_column];
            } else {
                weight_values[column] = 0.0;
            }
        }
        for (uint row = 0; row < 8; ++row) {
            for (uint column = 0; column < 4; ++column) {
                sums[row][column] +=
                    input_values[row] * weight_values[column];
            }
        }
    }
    for (uint row = 0; row < 8; ++row) {
        const uint output_row = row_base + row;
        if (output_row >= parameters.rows) continue;
        for (uint column = 0; column < 4; ++column) {
            const uint output_column = column_base + column;
            if (output_column < parameters.columns) {
                output_buffer.data[
                    (batch * parameters.rows + output_row) *
                        parameters.columns +
                    output_column] = sums[row][column];
            }
        }
    }
}
