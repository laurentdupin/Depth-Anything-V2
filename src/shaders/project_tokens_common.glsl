#if defined(NATIVE_FP16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Tokens {
    float data[];
} token_buffer;
#if defined(HALF_WEIGHT)
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    uint data[];
} weight_buffer;
#if defined(NATIVE_FP16)
float16_t read_weight(uint index) {
    const vec2 values =
        unpackHalf2x16(weight_buffer.data[index >> 1]);
    return float16_t((index & 1) == 0 ? values.x : values.y);
}
#else
float read_weight(uint index) {
    const vec2 values =
        unpackHalf2x16(weight_buffer.data[index >> 1]);
    return (index & 1) == 0 ? values.x : values.y;
}
#endif
#else
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
float read_weight(uint index) {
    return weight_buffer.data[index];
}
#endif
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;

layout(push_constant) uniform Parameters {
    uint width;
    uint height;
    uint embedding;
    uint output_channels;
} parameters;

#define INNER_STRIDE 17
#if defined(NATIVE_FP16)
shared float16_t token_tile[32 * INNER_STRIDE];
shared float16_t weight_tile[32 * INNER_STRIDE];
#define TILE_ZERO float16_t(0.0)
#else
shared float token_tile[32 * INNER_STRIDE];
shared float weight_tile[32 * INNER_STRIDE];
#define TILE_ZERO 0.0
#endif

void main() {
    const uint spatial = parameters.width * parameters.height;
    const uint channel_base =
        gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x * 4;
    const uint spatial_base =
        gl_WorkGroupID.y * 32 + gl_LocalInvocationID.y * 4;
    float sums[4][4];
    for (uint row = 0; row < 4; ++row) {
        for (uint column = 0; column < 4; ++column) {
            sums[row][column] = 0.0;
        }
    }
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    for (uint inner_base = 0;
         inner_base < parameters.embedding;
         inner_base += 16) {
        for (uint index = lane; index < 32 * 16; index += 64) {
            const uint tile_row = index / 16;
            const uint inner = inner_base + index % 16;
            const uint output_spatial =
                gl_WorkGroupID.y * 32 + tile_row;
            token_tile[tile_row * INNER_STRIDE + index % 16] =
                output_spatial < spatial &&
                    inner < parameters.embedding
                ?
#if defined(NATIVE_FP16)
                    float16_t(token_buffer.data[
                    (output_spatial + 1) * parameters.embedding + inner])
#else
                    token_buffer.data[
                    (output_spatial + 1) * parameters.embedding + inner]
#endif
                : TILE_ZERO;
        }
        for (uint index = lane; index < 32 * 16; index += 64) {
            const uint tile_column = index / 16;
            const uint inner = inner_base + index % 16;
            const uint output_channel =
                gl_WorkGroupID.x * 32 + tile_column;
            weight_tile[tile_column * INNER_STRIDE + index % 16] =
                output_channel < parameters.output_channels &&
                    inner < parameters.embedding
                ? read_weight(
                    output_channel * parameters.embedding + inner)
                : TILE_ZERO;
        }
        barrier();
        const uint inner_count =
            min(16, parameters.embedding - inner_base);
        for (uint inner = 0; inner < inner_count; ++inner) {
#if defined(NATIVE_FP16)
            float16_t token_values[4];
            float16_t weight_values[4];
#else
            float token_values[4];
            float weight_values[4];
#endif
            for (uint row = 0; row < 4; ++row) {
                token_values[row] = token_tile[
                    (gl_LocalInvocationID.y * 4 + row) *
                        INNER_STRIDE + inner];
            }
            for (uint column = 0; column < 4; ++column) {
                weight_values[column] = weight_tile[
                    (gl_LocalInvocationID.x * 4 + column) *
                        INNER_STRIDE + inner];
            }
            for (uint row = 0; row < 4; ++row) {
                for (uint column = 0; column < 4; ++column) {
                    sums[row][column] += float(
                        token_values[row] * weight_values[column]);
                }
            }
        }
        barrier();
    }
    for (uint row = 0; row < 4; ++row) {
        const uint output_spatial = spatial_base + row;
        if (output_spatial >= spatial) continue;
        for (uint column = 0; column < 4; ++column) {
            const uint output_channel = channel_base + column;
            if (output_channel < parameters.output_channels) {
                output_buffer.data[
                    output_channel * spatial + output_spatial] =
                    sums[row][column] + bias_buffer.data[output_channel];
            }
        }
    }
}
