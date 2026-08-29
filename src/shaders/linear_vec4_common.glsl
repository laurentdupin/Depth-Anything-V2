#if defined(COMPACT_TILE)
#define TILE_ROWS 64
#define ROWS_PER_LANE 8
#define INVOCATIONS 128
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
#else
#define TILE_ROWS 56
#define ROWS_PER_LANE 7
#define INVOCATIONS 128
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
#endif

#if defined(FP16_OUTPUT)
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    f16vec4 data[];
} output_buffer;
#else
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
#endif
#if defined(FP16_ARITHMETIC)
#define DAV2_VEC4 f16vec4
layout(set = 0, binding = 1, std430) readonly buffer Input {
    f16vec4 data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    f16vec4 data[];
} weight_buffer;
f16vec4 read_weight4(uint vector_index) {
    return weight_buffer.data[vector_index];
}
#else
#define DAV2_VEC4 vec4
layout(set = 0, binding = 1, std430) readonly buffer Input {
    vec4 data[];
} input_buffer;
#if defined(HALF_WEIGHT)
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    uint data[];
} weight_buffer;
vec4 read_weight4(uint vector_index) {
    const uint packed_index = vector_index * 2;
    return vec4(
        unpackHalf2x16(weight_buffer.data[packed_index]),
        unpackHalf2x16(weight_buffer.data[packed_index + 1]));
}
#else
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    vec4 data[];
} weight_buffer;
vec4 read_weight4(uint vector_index) {
    return weight_buffer.data[vector_index];
}
#endif
#endif
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
#if defined(FUSED_RESIDUAL)
layout(set = 0, binding = 4, std430) readonly buffer Residual {
    float data[];
} residual_buffer;
layout(set = 0, binding = 5, std430) readonly buffer Scale {
    float data[];
} scale_buffer;
#endif

layout(push_constant) uniform Parameters {
    uint rows;
    uint input_columns;
    uint output_columns;
} parameters;

#if defined(FUSED_GELU)
float exact_gelu(float value) {
    const float x = value * 0.7071067811865475;
    const float magnitude = abs(x);
    const float t = 1.0 / (1.0 + 0.5 * magnitude);
    const float polynomial =
        -1.26551223 + t * (1.00002368 + t * (0.37409196 +
        t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 +
        t * (-1.13520398 + t * (1.48851587 +
        t * (-0.82215223 + t * 0.17087277))))))));
    const float erfc_magnitude =
        t * exp(-magnitude * magnitude + polynomial);
    const float erf_value =
        x >= 0.0 ? 1.0 - erfc_magnitude : erfc_magnitude - 1.0;
    return 0.5 * value * (1.0 + erf_value);
}
#endif

#ifndef K_VECTORS
#define K_VECTORS 4
#endif
#define K_STRIDE (K_VECTORS + 1)
shared DAV2_VEC4 input_tile[TILE_ROWS * K_STRIDE];
shared DAV2_VEC4 weight_tile[64 * K_STRIDE];

void main() {
    const uint column_base =
        gl_WorkGroupID.x * 64 + gl_LocalInvocationID.x * 4;
    const uint row_base =
        gl_WorkGroupID.y * TILE_ROWS +
        gl_LocalInvocationID.y * ROWS_PER_LANE;
    float sums[ROWS_PER_LANE][4];
    for (uint row = 0; row < ROWS_PER_LANE; ++row) {
        for (uint column = 0; column < 4; ++column) {
            sums[row][column] = 0.0;
        }
    }
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    const uint input_vectors = parameters.input_columns / 4;
    for (uint inner_vector_base = 0;
         inner_vector_base < input_vectors;
         inner_vector_base += K_VECTORS) {
        for (uint index = lane;
             index < TILE_ROWS * K_VECTORS;
             index += INVOCATIONS) {
            const uint tile_row = index / K_VECTORS;
            const uint inner_vector =
                inner_vector_base + index % K_VECTORS;
            const uint output_row =
                gl_WorkGroupID.y * TILE_ROWS + tile_row;
            input_tile[tile_row * K_STRIDE + index % K_VECTORS] =
                output_row < parameters.rows &&
                    inner_vector < input_vectors
                ? input_buffer.data[
                      output_row * input_vectors + inner_vector]
                : DAV2_VEC4(0.0);
        }
        for (uint index = lane;
             index < 64 * K_VECTORS;
             index += INVOCATIONS) {
            const uint tile_column = index / K_VECTORS;
            const uint inner_vector =
                inner_vector_base + index % K_VECTORS;
            const uint output_column =
                gl_WorkGroupID.x * 64 + tile_column;
            weight_tile[tile_column * K_STRIDE + index % K_VECTORS] =
                output_column < parameters.output_columns &&
                    inner_vector < input_vectors
                ? read_weight4(
                      output_column * input_vectors + inner_vector)
                : DAV2_VEC4(0.0);
        }
        barrier();
        const uint vector_count =
            min(K_VECTORS, input_vectors - inner_vector_base);
        for (uint inner_vector = 0;
             inner_vector < vector_count;
             ++inner_vector) {
            DAV2_VEC4 input_values[ROWS_PER_LANE];
            DAV2_VEC4 weight_values[4];
            for (uint row = 0; row < ROWS_PER_LANE; ++row) {
                input_values[row] = input_tile[
                    (gl_LocalInvocationID.y * ROWS_PER_LANE + row) *
                        K_STRIDE +
                    inner_vector];
            }
            for (uint column = 0; column < 4; ++column) {
                weight_values[column] = weight_tile[
                    (gl_LocalInvocationID.x * 4 + column) * K_STRIDE +
                    inner_vector];
            }
            for (uint row = 0; row < ROWS_PER_LANE; ++row) {
                for (uint column = 0; column < 4; ++column) {
                    sums[row][column] += float(
                        dot(input_values[row], weight_values[column]));
                }
            }
        }
        barrier();
    }
    for (uint row = 0; row < ROWS_PER_LANE; ++row) {
        const uint output_row = row_base + row;
        if (output_row >= parameters.rows) {
            continue;
        }
#if defined(FP16_OUTPUT)
        f16vec4 packed_output = f16vec4(0.0);
#endif
        for (uint column = 0; column < 4; ++column) {
            const uint output_column = column_base + column;
            if (output_column < parameters.output_columns) {
                float value =
                    sums[row][column] + bias_buffer.data[output_column];
#if defined(FUSED_GELU)
                value = exact_gelu(value);
#endif
#if defined(FUSED_RESIDUAL)
                value =
                    residual_buffer.data[
                        output_row * parameters.output_columns +
                        output_column] +
                    value * scale_buffer.data[output_column];
#endif
#if defined(FP16_OUTPUT)
                packed_output[column] = float16_t(value);
#else
                output_buffer.data[
                    output_row * parameters.output_columns + output_column] =
                    value;
#endif
            }
        }
#if defined(FP16_OUTPUT)
        if (column_base < parameters.output_columns) {
            output_buffer.data[
                output_row * (parameters.output_columns / 4) +
                column_base / 4] = packed_output;
        }
#endif
    }
}
