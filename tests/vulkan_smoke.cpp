#include "elementwise_spv.h"
#include "gpu_model.h"
#include "operators.h"
#include "vulkan.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

void expect_near(float actual, float expected, float tolerance) {
    assert(std::abs(actual - expected) <= tolerance);
}

}  // namespace

int main() {
    static constexpr char crc_text[] = "123456789";
    assert(dav2::crc32(crc_text, 9) == 0xcbf43926u);

    dav2::VulkanContext context(0);
    const std::vector<float> input{
        -4.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 7.0f, 12.0f,
    };
    auto gpu_input = context.create_device_buffer(input.size() * sizeof(float));
    auto gpu_output = context.create_device_buffer(input.size() * sizeof(float));
    context.upload(gpu_input, input.data(), input.size() * sizeof(float));

    auto pipeline = context.create_pipeline(
        dav2_elementwise_spv,
        dav2_elementwise_spv_size,
        2,
        16);
    struct Parameters {
        std::uint32_t count;
        float scale;
        float bias;
        std::uint32_t relu;
    } parameters{
        static_cast<std::uint32_t>(input.size()),
        1.5f,
        -0.25f,
        1,
    };
    context.dispatch(
        pipeline,
        {&gpu_input, &gpu_output},
        &parameters,
        sizeof(parameters),
        1);

    std::vector<float> output(input.size());
    context.download(
        gpu_output, output.data(), output.size() * sizeof(float));
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float expected =
            std::max(input[index] * parameters.scale + parameters.bias, 0.0f);
        assert(std::abs(output[index] - expected) == 0.0f);
    }

    dav2::VulkanOperators operators(context);
    const std::uint32_t rows = 3;
    const std::uint32_t input_columns = 19;
    const std::uint32_t output_columns = 7;
    std::vector<float> matrix_input(rows * input_columns);
    std::vector<float> matrix_weight(output_columns * input_columns);
    std::vector<float> matrix_bias(output_columns);
    for (std::size_t index = 0; index < matrix_input.size(); ++index) {
        matrix_input[index] =
            (static_cast<int>(index % 13) - 6) * 0.03125f;
    }
    for (std::size_t index = 0; index < matrix_weight.size(); ++index) {
        matrix_weight[index] =
            (static_cast<int>(index % 11) - 5) * 0.015625f;
    }
    for (std::size_t index = 0; index < matrix_bias.size(); ++index) {
        matrix_bias[index] = static_cast<float>(index) * 0.0625f;
    }
    auto gpu_matrix_input =
        context.create_device_buffer(matrix_input.size() * sizeof(float));
    auto gpu_matrix_weight =
        context.create_device_buffer(matrix_weight.size() * sizeof(float));
    auto gpu_matrix_bias =
        context.create_device_buffer(matrix_bias.size() * sizeof(float));
    auto gpu_matrix_output = context.create_device_buffer(
        rows * output_columns * sizeof(float));
    context.upload(
        gpu_matrix_input,
        matrix_input.data(),
        matrix_input.size() * sizeof(float));
    context.upload(
        gpu_matrix_weight,
        matrix_weight.data(),
        matrix_weight.size() * sizeof(float));
    context.upload(
        gpu_matrix_bias,
        matrix_bias.data(),
        matrix_bias.size() * sizeof(float));
    operators.linear(
        gpu_matrix_output,
        gpu_matrix_input,
        gpu_matrix_weight,
        gpu_matrix_bias,
        rows,
        input_columns,
        output_columns,
        false);
    std::vector<float> matrix_output(rows * output_columns);
    context.download(
        gpu_matrix_output,
        matrix_output.data(),
        matrix_output.size() * sizeof(float));
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < output_columns; ++column) {
            float expected = 0.0f;
            for (std::uint32_t inner = 0; inner < input_columns; ++inner) {
                expected += matrix_input[row * input_columns + inner] *
                    matrix_weight[column * input_columns + inner];
            }
            expected += matrix_bias[column];
            expect_near(
                matrix_output[row * output_columns + column],
                expected,
                1.0e-6f);
        }
    }
    operators.linear(
        gpu_matrix_output,
        gpu_matrix_input,
        gpu_matrix_weight,
        gpu_matrix_bias,
        rows,
        input_columns,
        output_columns,
        true);
    context.download(
        gpu_matrix_output,
        matrix_output.data(),
        matrix_output.size() * sizeof(float));
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < output_columns; ++column) {
            float expected = 0.0f;
            for (std::uint32_t inner = 0; inner < input_columns; ++inner) {
                expected += matrix_input[row * input_columns + inner] *
                    matrix_weight[column * input_columns + inner];
            }
            expected += matrix_bias[column];
            const float cube = expected * expected * expected;
            const float inner = 0.7978845608028654f *
                (expected + 0.044715f * cube);
            expected = 0.5f * expected *
                (1.0f + std::tanh(std::clamp(inner, -15.0f, 15.0f)));
            expect_near(
                matrix_output[row * output_columns + column],
                expected,
                1.0e-6f);
        }
    }

    const std::uint32_t norm_rows = 4;
    const std::uint32_t norm_columns = 17;
    std::vector<float> norm_input(norm_rows * norm_columns);
    std::vector<float> norm_weight(norm_columns);
    std::vector<float> norm_bias(norm_columns);
    for (std::size_t index = 0; index < norm_input.size(); ++index) {
        norm_input[index] =
            (static_cast<int>(index % 23) - 10) * 0.125f;
    }
    std::fill(norm_weight.begin(), norm_weight.end(), 1.0f);
    std::fill(norm_bias.begin(), norm_bias.end(), 0.0f);
    auto gpu_norm_input =
        context.create_device_buffer(norm_input.size() * sizeof(float));
    auto gpu_norm_weight =
        context.create_device_buffer(norm_weight.size() * sizeof(float));
    auto gpu_norm_bias =
        context.create_device_buffer(norm_bias.size() * sizeof(float));
    auto gpu_norm_output =
        context.create_device_buffer(norm_input.size() * sizeof(float));
    context.upload(
        gpu_norm_input, norm_input.data(), norm_input.size() * sizeof(float));
    context.upload(
        gpu_norm_weight,
        norm_weight.data(),
        norm_weight.size() * sizeof(float));
    context.upload(
        gpu_norm_bias, norm_bias.data(), norm_bias.size() * sizeof(float));
    operators.layer_norm(
        gpu_norm_output,
        gpu_norm_input,
        gpu_norm_weight,
        gpu_norm_bias,
        norm_rows,
        norm_columns,
        1.0e-6f);
    std::vector<float> norm_output(norm_input.size());
    context.download(
        gpu_norm_output,
        norm_output.data(),
        norm_output.size() * sizeof(float));
    for (std::uint32_t row = 0; row < norm_rows; ++row) {
        float sum = 0.0f;
        float sum_of_squares = 0.0f;
        for (std::uint32_t column = 0; column < norm_columns; ++column) {
            const float value = norm_input[row * norm_columns + column];
            sum += value;
            sum_of_squares += value * value;
        }
        const float mean = sum / norm_columns;
        const float variance =
            std::max(sum_of_squares / norm_columns - mean * mean, 0.0f);
        const float inverse = 1.0f / std::sqrt(variance + 1.0e-6f);
        for (std::uint32_t column = 0; column < norm_columns; ++column) {
            expect_near(
                norm_output[row * norm_columns + column],
                (norm_input[row * norm_columns + column] - mean) * inverse,
                1.0e-5f);
        }
    }

    const std::uint32_t scaled_columns = 5;
    const std::uint32_t scaled_count = 15;
    std::vector<float> residual(scaled_count);
    std::vector<float> addend(scaled_count);
    std::vector<float> scale(scaled_columns);
    for (std::uint32_t index = 0; index < scaled_count; ++index) {
        residual[index] = static_cast<float>(index) * 0.125f;
        addend[index] = static_cast<float>(index + 2) * -0.0625f;
    }
    for (std::uint32_t index = 0; index < scaled_columns; ++index) {
        scale[index] = static_cast<float>(index + 1) * 0.25f;
    }
    auto gpu_residual =
        context.create_device_buffer(residual.size() * sizeof(float));
    auto gpu_addend =
        context.create_device_buffer(addend.size() * sizeof(float));
    auto gpu_scale =
        context.create_device_buffer(scale.size() * sizeof(float));
    auto gpu_scaled_output =
        context.create_device_buffer(residual.size() * sizeof(float));
    context.upload(
        gpu_residual, residual.data(), residual.size() * sizeof(float));
    context.upload(gpu_addend, addend.data(), addend.size() * sizeof(float));
    context.upload(gpu_scale, scale.data(), scale.size() * sizeof(float));
    operators.add_scaled(
        gpu_scaled_output,
        gpu_residual,
        gpu_addend,
        gpu_scale,
        scaled_count,
        scaled_columns);
    std::vector<float> scaled_output(scaled_count);
    context.download(
        gpu_scaled_output,
        scaled_output.data(),
        scaled_output.size() * sizeof(float));
    for (std::uint32_t index = 0; index < scaled_count; ++index) {
        expect_near(
            scaled_output[index],
            residual[index] +
                addend[index] * scale[index % scaled_columns],
            0.0f);
    }

    const std::uint32_t attention_tokens = 5;
    const std::uint32_t attention_heads = 2;
    const std::uint32_t attention_embedding = attention_heads * 64;
    std::vector<float> qkv(
        attention_tokens * attention_embedding * 3);
    for (std::size_t index = 0; index < qkv.size(); ++index) {
        qkv[index] =
            (static_cast<int>(index % 29) - 14) * 0.0078125f;
    }
    const std::size_t attention_elements =
        attention_tokens * attention_embedding;
    auto gpu_qkv =
        context.create_device_buffer(qkv.size() * sizeof(float));
    auto gpu_query =
        context.create_device_buffer(attention_elements * sizeof(float));
    auto gpu_key =
        context.create_device_buffer(attention_elements * sizeof(float));
    auto gpu_value =
        context.create_device_buffer(attention_elements * sizeof(float));
    auto gpu_attention =
        context.create_device_buffer(attention_elements * sizeof(float));
    auto gpu_merged =
        context.create_device_buffer(attention_elements * sizeof(float));
    context.upload(gpu_qkv, qkv.data(), qkv.size() * sizeof(float));
    operators.split_qkv(
        gpu_query,
        gpu_key,
        gpu_value,
        gpu_qkv,
        attention_tokens,
        attention_heads);
    operators.attention_head64(
        gpu_attention,
        gpu_query,
        gpu_key,
        gpu_value,
        attention_tokens,
        attention_heads);
    operators.merge_heads(
        gpu_merged,
        gpu_attention,
        attention_tokens,
        attention_heads);
    std::vector<float> merged(attention_elements);
    context.download(
        gpu_merged, merged.data(), merged.size() * sizeof(float));
    for (std::uint32_t token = 0; token < attention_tokens; ++token) {
        for (std::uint32_t head = 0; head < attention_heads; ++head) {
            std::vector<float> scores(attention_tokens);
            float maximum = -std::numeric_limits<float>::max();
            for (std::uint32_t source = 0;
                 source < attention_tokens;
                 ++source) {
                float score = 0.0f;
                for (std::uint32_t feature = 0; feature < 64; ++feature) {
                    const std::size_t query_index =
                        token * attention_embedding * 3 +
                        head * 64 + feature;
                    const std::size_t key_index =
                        source * attention_embedding * 3 +
                        attention_embedding + head * 64 + feature;
                    score += qkv[query_index] * 0.125f *
                        qkv[key_index];
                }
                scores[source] = score;
                maximum = std::max(maximum, score);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (std::uint32_t feature = 0; feature < 64; ++feature) {
                float expected = 0.0f;
                for (std::uint32_t source = 0;
                     source < attention_tokens;
                     ++source) {
                    const std::size_t value_index =
                        source * attention_embedding * 3 +
                        attention_embedding * 2 +
                        head * 64 + feature;
                    expected += scores[source] / denominator *
                        qkv[value_index];
                }
                expect_near(
                    merged[token * attention_embedding +
                        head * 64 + feature],
                    expected,
                    2.0e-6f);
            }
        }
    }
    std::cout << context.device_name() << '\n';
    return 0;
}
