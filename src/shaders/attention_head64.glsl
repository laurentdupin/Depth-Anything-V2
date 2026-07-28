#version 450 core

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Query {
    float data[];
} query_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Key {
    float data[];
} key_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Value {
    float data[];
} value_buffer;

layout(push_constant) uniform Parameters {
    uint tokens;
    uint heads;
} parameters;

const float NEGATIVE_INFINITY = -3.402823466e+38;
const float MINIMUM_DENOMINATOR = 1.0e-20;
const uint QUERY_ROWS = 4;
shared float score_partials[QUERY_ROWS * 64];

void main() {
    const uint lane = gl_LocalInvocationID.x;
    const uint query_base = gl_WorkGroupID.y * QUERY_ROWS;
    const uint head = gl_WorkGroupID.z;
    if (query_base >= parameters.tokens || head >= parameters.heads) {
        return;
    }

    bool query_valid[QUERY_ROWS];
    float query_value[QUERY_ROWS];
    float accumulator[QUERY_ROWS];
    float row_maximum[QUERY_ROWS];
    float row_denominator[QUERY_ROWS];
    for (uint row = 0; row < QUERY_ROWS; ++row) {
        const uint query_token = query_base + row;
        query_valid[row] = query_token < parameters.tokens;
        query_value[row] = query_valid[row]
            ? query_buffer.data[
                  (head * parameters.tokens + query_token) * 64 + lane]
            : 0.0;
        accumulator[row] = 0.0;
        row_maximum[row] = NEGATIVE_INFINITY;
        row_denominator[row] = 0.0;
    }

    for (uint source = 0; source < parameters.tokens; ++source) {
        const uint source_index =
            (head * parameters.tokens + source) * 64 + lane;
        const float key_value = key_buffer.data[source_index];
        for (uint row = 0; row < QUERY_ROWS; ++row) {
            score_partials[row * 64 + lane] =
                query_valid[row] ? query_value[row] * key_value : 0.0;
        }
        barrier();
        for (uint offset = 32; offset > 0; offset /= 2) {
            if (lane < offset) {
                for (uint row = 0; row < QUERY_ROWS; ++row) {
                    score_partials[row * 64 + lane] +=
                        score_partials[row * 64 + lane + offset];
                }
            }
            barrier();
        }

        const float value = value_buffer.data[source_index];
        for (uint row = 0; row < QUERY_ROWS; ++row) {
            if (query_valid[row]) {
                const float score = score_partials[row * 64];
                const float new_maximum = max(row_maximum[row], score);
                const float previous_scale =
                    exp(row_maximum[row] - new_maximum);
                const float current_scale = exp(score - new_maximum);
                row_denominator[row] =
                    row_denominator[row] * previous_scale + current_scale;
                row_maximum[row] = new_maximum;
                accumulator[row] =
                    accumulator[row] * previous_scale +
                    current_scale * value;
            }
        }
        barrier();
    }

    for (uint row = 0; row < QUERY_ROWS; ++row) {
        if (query_valid[row]) {
            const uint query_token = query_base + row;
            output_buffer.data[
                (head * parameters.tokens + query_token) * 64 + lane] =
                accumulator[row] /
                max(row_denominator[row], MINIMUM_DENOMINATOR);
        }
    }
}
