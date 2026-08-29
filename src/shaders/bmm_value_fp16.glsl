#version 450 core
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Scores { uint data[]; } score_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Qkv { float data[]; } qkv_buffer;
layout(push_constant) uniform Parameters { uint tokens; uint heads; } parameters;

#define INNER_STRIDE 17
shared float16_t score_tile[64 * INNER_STRIDE];
shared float16_t value_tile[32 * INNER_STRIDE];

float16_t read_score(uint row, uint column, uint head) {
    const uint words = (parameters.tokens + 1u) >> 1u;
    const vec2 pair = unpackHalf2x16(score_buffer.data[(head * parameters.tokens + row) * words + column / 2u]);
    return float16_t((column & 1u) == 0u ? pair.x : pair.y);
}

void main() {
    const uint feature_base = gl_WorkGroupID.x * 32u + gl_LocalInvocationID.x * 4u;
    const uint row_base = gl_WorkGroupID.y * 64u + gl_LocalInvocationID.y * 8u;
    const uint head = gl_GlobalInvocationID.z;
    const uint embedding = parameters.heads * 64u;
    float sums[8][4];
    for (uint row = 0u; row < 8u; ++row) {
        for (uint column = 0u; column < 4u; ++column) sums[row][column] = 0.0;
    }
    const uint lane = gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    for (uint inner_base = 0u; inner_base < parameters.tokens; inner_base += 16u) {
        for (uint index = lane; index < 64u * 16u; index += 64u) {
            const uint tile_row = index / 16u;
            const uint source = inner_base + index % 16u;
            const uint row = gl_WorkGroupID.y * 64u + tile_row;
            score_tile[tile_row * INNER_STRIDE + index % 16u] =
                head < parameters.heads && row < parameters.tokens && source < parameters.tokens
                ? read_score(row, source, head) : float16_t(0.0);
        }
        for (uint index = lane; index < 32u * 16u; index += 64u) {
            const uint tile_column = index / 16u;
            const uint source = inner_base + index % 16u;
            const uint feature = gl_WorkGroupID.x * 32u + tile_column;
            value_tile[tile_column * INNER_STRIDE + index % 16u] =
                head < parameters.heads && feature < 64u && source < parameters.tokens
                ? float16_t(qkv_buffer.data[source * embedding * 3u + embedding * 2u + head * 64u + feature])
                : float16_t(0.0);
        }
        barrier();
        const uint inner_count = min(16u, parameters.tokens - inner_base);
        for (uint inner = 0u; inner < inner_count; ++inner) {
            float16_t score_values[8];
            float16_t value_values[4];
            for (uint row = 0u; row < 8u; ++row) {
                score_values[row] = score_tile[(gl_LocalInvocationID.y * 8u + row) * INNER_STRIDE + inner];
            }
            for (uint column = 0u; column < 4u; ++column) {
                value_values[column] = value_tile[(gl_LocalInvocationID.x * 4u + column) * INNER_STRIDE + inner];
            }
            for (uint row = 0u; row < 8u; ++row) {
                for (uint column = 0u; column < 4u; ++column) {
                    sums[row][column] += float(score_values[row] * value_values[column]);
                }
            }
        }
        barrier();
    }
    for (uint row = 0u; row < 8u; ++row) {
        const uint output_row = row_base + row;
        if (output_row >= parameters.tokens || head >= parameters.heads) continue;
        for (uint column = 0u; column < 4u; ++column) {
            const uint feature = feature_base + column;
            if (feature < 64u) {
                output_buffer.data[(output_row * parameters.heads + head) * 64u + feature] = sums[row][column];
            }
        }
    }
}
