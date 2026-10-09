#include "vulkan.h"
#include <inferbridge/native_harness_vulkan_subgroup.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "copy_words_spv.h"
#include "operators.h"

template <typename T>
struct Comparison {
    std::vector<T> original, candidate;
    double original_ms, candidate_ms;
};

template <typename T, typename Parameters>
Comparison<T> compare(dav2::VulkanContext& context, dav2::VulkanPipeline& original,
                      dav2::VulkanPipeline& candidate, dav2::VulkanBuffer& output,
                      const std::vector<const dav2::VulkanBuffer*>& buffers,
                      const Parameters& parameters, std::array<unsigned, 3> original_groups,
                      std::array<unsigned, 3> candidate_groups,
                      const std::function<void(unsigned)>& record = {}) {
    Comparison<T> result;
    result.original.resize(static_cast<std::size_t>(output.size()) / sizeof(T));
    result.candidate.resize(result.original.size());
    std::vector<double> timings[2];
    for (unsigned round = 0; round < 8; ++round) {
        for (unsigned order = 0; order < 2; ++order) {
            const unsigned which = (round + order) % 2;
            auto& pipeline = which == 0 ? original : candidate;
            const auto groups = which == 0 ? original_groups : candidate_groups;
            const auto start = std::chrono::steady_clock::now();
            context.batch([&] {
                for (unsigned repeat = 0; repeat < 20; ++repeat)
                    if (record)
                        record(which);
                    else
                        context.dispatch(pipeline, buffers, &parameters, sizeof(parameters),
                                         groups[0], groups[1], groups[2]);
            });
            if (round > 0)
                timings[which].push_back(std::chrono::duration<double, std::milli>(
                                             std::chrono::steady_clock::now() - start)
                                             .count() /
                                         20);
            auto& destination = which == 0 ? result.original : result.candidate;
            context.download(output, destination.data(), destination.size() * sizeof(T));
        }
    }
    for (auto& times : timings) std::sort(times.begin(), times.end());
    result.original_ms = timings[0][timings[0].size() / 2];
    result.candidate_ms = timings[1][timings[1].size() / 2];
    return result;
}

void attention_score(dav2::VulkanContext& context, const std::vector<std::uint32_t>& old_code,
                     const std::vector<std::uint32_t>& new_code) {
    auto original = context.create_pipeline(old_code.data(), old_code.size() * 4, 2, 8);
    auto tiled = context.create_pipeline(new_code.data(), new_code.size() * 4, 2, 8);
    for (const auto dimensions : {std::array<unsigned, 2>{17, 2}, {401, 6}, {1370, 16}}) {
        struct Parameters {
            unsigned tokens, heads;
        } p{dimensions[0], dimensions[1]};
        const unsigned embedding = p.heads * 64, words = (p.tokens + 1) / 2;
        std::vector<float> qkv(std::size_t(p.tokens) * embedding * 3);
        for (std::size_t n = 0; n < qkv.size(); ++n) qkv[n] = float(int(n % 17) - 8) * 0.125f;
        auto input = context.create_device_buffer(qkv.size() * 4);
        auto output = context.create_device_buffer(std::uint64_t(p.heads) * p.tokens * words * 4);
        context.upload(input, qkv.data(), qkv.size() * 4);
        const auto result =
            compare<std::uint32_t>(context, original, tiled, output, {&output, &input}, p,
                                   {(p.tokens + 15) / 16, (p.tokens + 15) / 16, p.heads},
                                   {(p.tokens + 31) / 32, (p.tokens + 63) / 64, p.heads});
        if (result.original != result.candidate)
            throw std::runtime_error("attention score parity failed");
        if (p.tokens % 2 != 0)
            for (unsigned h = 0; h < p.heads; ++h)
                for (unsigned row = 0; row < p.tokens; ++row)
                    if (result.candidate[(h * p.tokens + row) * words + words - 1] >> 16)
                        throw std::runtime_error("odd attention tail was not zero-filled");
        std::cout << "ATTENTION_SCORE_OK tokens=" << p.tokens << " heads=" << p.heads
                  << " original_ms=" << result.original_ms << " tiled_ms=" << result.candidate_ms
                  << " bit_exact=true\n";
    }
}

std::vector<std::uint32_t> read_spirv(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto bytes = file.tellg();
    if (!file || bytes <= 0 || bytes % 4 != 0) throw std::runtime_error("invalid SPIR-V file");
    std::vector<std::uint32_t> words(static_cast<std::size_t>(bytes) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), bytes);
    if (!file) throw std::runtime_error("failed to read SPIR-V file");
    return words;
}

float half_value(std::uint16_t bits) {
    const unsigned exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    const float value = exponent == 0
                            ? std::ldexp(float(mantissa), -24)
                            : std::ldexp(1.0f + float(mantissa) / 1024.0f, int(exponent) - 15);
    return bits & 0x8000 ? -value : value;
}

void softmax(dav2::VulkanContext& context, const std::vector<std::uint32_t>& old_code,
             const std::vector<std::uint32_t>& new_code, bool half) {
    const std::vector<VkDescriptorType> types(half ? 1 : 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    const auto access =
        half ? std::vector<VkAccessFlags>{VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT}
             : std::vector<VkAccessFlags>{VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    auto original = context.create_pipeline(old_code.data(), old_code.size() * 4, types, access, 8);
    auto candidate =
        context.create_pipeline(new_code.data(), new_code.size() * 4, types, access, 8);
    auto copy = context.create_pipeline(dav2_copy_words_spv, dav2_copy_words_spv_size, 2, 4);
    constexpr std::uint16_t packed[7] = {0xb600, 0xb400, 0xb000, 0, 0x3000, 0x3400, 0x3600};
    for (unsigned columns : {1u, 17u, 401u, 1370u}) {
        struct Parameters {
            unsigned rows, columns;
        } p{columns < 128 ? 19u : columns * 6, columns};
        const unsigned words = half ? (columns + 1) / 2 : columns;
        std::vector<float> input(std::size_t(p.rows) * columns);
        std::vector<std::uint32_t> half_input(std::size_t(p.rows) * words);
        for (unsigned row = 0; row < p.rows; ++row)
            for (unsigned col = 0; col < columns; ++col) {
                const unsigned index = row * columns + col;
                input[index] = half_value(packed[index % 7]);
                half_input[row * words + col / 2] |= std::uint32_t(packed[index % 7])
                                                     << (col % 2 * 16);
            }
        auto seed = context.create_device_buffer(std::uint64_t(p.rows) * words * 4);
        auto output = context.create_device_buffer(seed.size());
        if (half)
            context.upload(seed, half_input.data(), half_input.size() * 4);
        else
            context.upload(seed, input.data(), input.size() * 4);
        const unsigned count = p.rows * words;
        const auto result = compare<std::uint32_t>(
            context, original, candidate, output, {}, p, {p.rows, 1, 1}, {p.rows, 1, 1},
            [&](unsigned which) {
                context.dispatch(copy, {&output, &seed}, &count, 4, (count + 255) / 256);
                auto& pipeline = which == 0 ? original : candidate;
                if (half)
                    context.dispatch(pipeline, {&output}, &p, 8, p.rows);
                else
                    context.dispatch(pipeline, {&output, &output}, &p, 8, p.rows);
            });
        double error = 0, magnitude = 0, cpu_error = 0;
        for (unsigned row = 0; row < p.rows; ++row) {
            double denominator = 0;
            for (unsigned col = 0; col < columns; ++col)
                denominator += std::exp(double(input[row * columns + col]));
            for (unsigned col = 0; col < columns; ++col) {
                float before, after;
                if (half) {
                    before = half_value(
                        std::uint16_t(result.original[row * words + col / 2] >> (col % 2 * 16)));
                    after = half_value(
                        std::uint16_t(result.candidate[row * words + col / 2] >> (col % 2 * 16)));
                } else {
                    const auto old_bits = result.original[row * words + col],
                               new_bits = result.candidate[row * words + col];
                    std::memcpy(&before, &old_bits, 4);
                    std::memcpy(&after, &new_bits, 4);
                }
                if (!std::isfinite(after))
                    throw std::runtime_error("non-finite subgroup softmax output");
                error += std::abs(double(before) - after);
                magnitude += std::abs(before);
                cpu_error = std::max(
                    cpu_error,
                    std::abs(after - std::exp(double(input[row * columns + col])) / denominator));
            }
            if (half && columns % 2 && result.candidate[row * words + words - 1] >> 16)
                throw std::runtime_error("softmax odd tail was not zero-filled");
        }
        if (error / std::max(magnitude, 1e-12) > 0.001 || cpu_error > (half ? 0.001 : 3e-6))
            throw std::runtime_error("subgroup softmax numerical gate failed");
        std::cout << "SOFTMAX_OK half=" << half << " columns=" << columns
                  << " original_ms=" << result.original_ms << " subgroup_ms=" << result.candidate_ms
                  << " relative_l1=" << error / std::max(magnitude, 1e-12)
                  << " cpu_max_error=" << cpu_error << '\n';
    }
}

void flash_attention(dav2::VulkanContext& context, const std::vector<std::uint32_t>& new_code) {
    dav2::VulkanOperators operators(context);
    auto fused = context.create_pipeline(new_code.data(), new_code.size() * 4, 2, 8);
    for (const auto shape : {std::array<unsigned, 2>{17, 2}, {401, 6}, {1370, 16}}) {
        struct Parameters {
            unsigned tokens, heads;
        } p{shape[0], shape[1]};
        std::vector<float> qkv(std::size_t(p.tokens) * p.heads * 64 * 3);
        for (std::size_t n = 0; n < qkv.size(); ++n) qkv[n] = std::sin(float(n) * 0.017f) * 1.5f;
        auto input = context.create_device_buffer(qkv.size() * 4);
        auto output = context.create_device_buffer(std::uint64_t(p.tokens) * p.heads * 64 * 4);
        auto scratch = context.create_device_buffer(std::uint64_t(p.heads) * p.tokens *
                                                    ((p.tokens + 1) / 2) * 4);
        context.upload(input, qkv.data(), qkv.size() * 4);
        const auto result = compare<float>(
            context, fused, fused, output, {&output, &input}, p, {1, 1, 1}, {1, 1, 1},
            [&](unsigned which) {
                if (which == 0)
                    operators.attention_head64(output, input, p.tokens, p.heads, &scratch,
                                               inferbridge::native::Precision::fp16);
                else
                    context.dispatch(fused, {&output, &input}, &p, sizeof(p), (p.tokens + 15) / 16,
                                     p.heads);
            });
        double error = 0, magnitude = 0, maximum = 0;
        for (std::size_t n = 0; n < result.original.size(); ++n) {
            if (!std::isfinite(result.candidate[n]))
                throw std::runtime_error("non-finite fused attention output");
            const double delta = std::abs(double(result.original[n]) - result.candidate[n]);
            error += delta;
            magnitude += std::abs(result.original[n]);
            maximum = std::max(maximum, delta);
        }
        const double relative = error / std::max(magnitude, 1e-12);
        std::cout << "FLASH_ATTENTION_OK tokens=" << p.tokens << " heads=" << p.heads
                  << " original_ms=" << result.original_ms << " fused_ms=" << result.candidate_ms
                  << " relative_l1=" << relative << " max_error=" << maximum << '\n';
        if (relative > 0.0033 || maximum > 0.01)
            throw std::runtime_error("fused attention numerical gate failed");
    }
}

void convolution(dav2::VulkanContext& context, const std::vector<std::uint32_t>& old_code,
                 const std::vector<std::uint32_t>& new_code, bool implicit = false) {
    auto original = context.create_pipeline(old_code.data(), old_code.size() * 4, 4, 40);
    auto tiled = context.create_pipeline(new_code.data(), new_code.size() * 4, 4, 40);
    for (const auto shape :
         {std::array<unsigned, 4>{17, 19, 5, 10}, {80, 80, 64, 64}, {148, 148, 128, 128}}) {
        struct Parameters {
            unsigned iw, ih, ic, ow, oh, oc, kernel, stride;
            int padding;
            unsigned bias;
        } p{shape[0], shape[1], implicit ? (shape[2] + 3) / 4 * 4 : shape[2],
            shape[0], shape[1], shape[3],
            3,        1,        1,
            1};
        std::vector<float> input(std::size_t(p.iw) * p.ih * p.ic),
            weight(std::size_t(p.oc) * p.ic * 9), bias(p.oc);
        std::vector<std::uint16_t> half_weight(weight.size());
        for (std::size_t n = 0; n < input.size(); ++n) input[n] = float(int(n % 17) - 8) * 0.0625f;
        for (std::size_t n = 0; n < weight.size(); ++n) weight[n] = float(int(n % 7) - 3) * 0.125f;
        constexpr std::uint16_t packed[7] = {0xb600, 0xb400, 0xb000, 0, 0x3000, 0x3400, 0x3600};
        for (std::size_t n = 0; n < weight.size(); ++n) half_weight[n] = packed[n % 7];
        for (unsigned n = 0; n < p.oc; ++n) bias[n] = float(n % 5) * 0.01f;
        auto gi = context.create_device_buffer(input.size() * 4),
             gw = context.create_device_buffer(weight.size() * (implicit ? 2 : 4));
        auto gb = context.create_device_buffer(bias.size() * 4),
             output = context.create_device_buffer(std::uint64_t(p.ow) * p.oh * p.oc * 4);
        context.upload(gi, input.data(), input.size() * 4);
        if (implicit)
            context.upload(gw, half_weight.data(), half_weight.size() * 2);
        else
            context.upload(gw, weight.data(), weight.size() * 4);
        context.upload(gb, bias.data(), bias.size() * 4);
        const auto groups = std::array<unsigned, 3>{
            (p.ow + 7) / 8, (p.oh + 7) / 8, (p.oc + (implicit ? 3 : 7)) / (implicit ? 4 : 8)};
        const auto candidate_groups =
            implicit ? std::array<unsigned, 3>{(p.oc + 31) / 32, (p.ow * p.oh + 63) / 64, 1}
                     : groups;
        const auto result = compare<float>(context, original, tiled, output,
                                           {&output, &gi, &gw, &gb}, p, groups, candidate_groups);
        if (result.original != result.candidate)
            throw std::runtime_error("channel-blocked convolution parity failed");
        for (std::size_t n = 0; n < result.candidate.size();
             n += result.candidate.size() < 10000 ? 1 : 7919) {
            const unsigned x = unsigned(n % p.ow), y = unsigned(n / p.ow % p.oh),
                           oc = unsigned(n / (p.ow * p.oh));
            double expected = bias[oc];
            for (unsigned ic = 0; ic < p.ic; ++ic)
                for (int ky = 0; ky < 3; ++ky)
                    for (int kx = 0; kx < 3; ++kx) {
                        const int ix = int(x) + kx - 1, iy = int(y) + ky - 1;
                        if (ix >= 0 && iy >= 0 && ix < int(p.iw) && iy < int(p.ih))
                            expected +=
                                double(input[(ic * p.ih + unsigned(iy)) * p.iw + unsigned(ix)]) *
                                weight[(oc * p.ic + ic) * 9 + ky * 3 + kx];
                    }
            if (std::abs(expected - result.candidate[n]) > 1e-4)
                throw std::runtime_error("convolution CPU reference mismatch");
        }
        std::cout << "CONVOLUTION_OK size=" << p.iw << 'x' << p.ih << " ic=" << p.ic
                  << " oc=" << p.oc << " original_ms=" << result.original_ms
                  << " blocked_ms=" << result.candidate_ms << " bit_exact=true\n";
    }
}

void layer_norm(dav2::VulkanContext& context, const std::vector<std::uint32_t>& old_code,
                const std::vector<std::uint32_t>& new_code) {
    auto original = context.create_pipeline(old_code.data(), old_code.size() * 4, 4, 12);
    auto stable = context.create_pipeline(new_code.data(), new_code.size() * 4, 4, 12);
    for (unsigned columns : {70u, 384u, 1024u})
        for (float offset : {0.0f, 65536.0f}) {
            struct Parameters {
                unsigned rows, columns;
                float epsilon;
            } p{19, columns, 1e-6f};
            std::vector<float> input(std::size_t(p.rows) * p.columns), weight(columns, 1.0f),
                bias(columns, 0.0f);
            for (std::size_t n = 0; n < input.size(); ++n)
                input[n] = offset + float(int(n % 17) - 8) * 0.0625f;
            auto gi = context.create_device_buffer(input.size() * 4),
                 gw = context.create_device_buffer(weight.size() * 4);
            auto gb = context.create_device_buffer(bias.size() * 4),
                 output = context.create_device_buffer(input.size() * 4);
            context.upload(gi, input.data(), input.size() * 4);
            context.upload(gw, weight.data(), weight.size() * 4);
            context.upload(gb, bias.data(), bias.size() * 4);
            const auto result =
                compare<float>(context, original, stable, output, {&output, &gi, &gw, &gb}, p,
                               {p.rows, 1, 1}, {p.rows, 1, 1});
            double old_error = 0, new_error = 0;
            for (unsigned row = 0; row < p.rows; ++row) {
                double mean = 0, variance = 0;
                for (unsigned c = 0; c < columns; ++c) mean += input[row * columns + c];
                mean /= columns;
                for (unsigned c = 0; c < columns; ++c)
                    variance += std::pow(double(input[row * columns + c]) - mean, 2);
                variance /= columns;
                for (unsigned c = 0; c < columns; ++c) {
                    const auto index = row * columns + c;
                    const double expected = (input[index] - mean) / std::sqrt(variance + p.epsilon);
                    old_error = std::max(old_error, std::abs(expected - result.original[index]));
                    new_error = std::max(new_error, std::abs(expected - result.candidate[index]));
                    if (!std::isfinite(result.candidate[index]))
                        throw std::runtime_error("non-finite LayerNorm output");
                }
            }
            std::cout << "LAYER_NORM_OK columns=" << columns << " offset=" << offset
                      << " original_ms=" << result.original_ms
                      << " stable_ms=" << result.candidate_ms << " original_max_error=" << old_error
                      << " stable_max_error=" << new_error << '\n';
            if (new_error > 3e-5)
                throw std::runtime_error("stable LayerNorm CPU reference mismatch");
        }
}

void check_shader_classifier() {
    using inferbridge::native_harness::spirv_uses_cooperative_matrix;
    std::array<std::uint32_t, 7> module{0x07230203u, 0x00010600u, 0, 2, 0, (2u << 16) | 17u, 6022u};
    if (!spirv_uses_cooperative_matrix(module.data(), sizeof(module)))
        throw std::runtime_error("cooperative capability not found");
    if (spirv_uses_cooperative_matrix(module.data(), sizeof(module) - 4))
        throw std::runtime_error("truncated capability accepted");
    module[6] = 1;
    if (spirv_uses_cooperative_matrix(module.data(), sizeof(module)))
        throw std::runtime_error("portable shader misclassified");
    module[5] = 17;
    if (spirv_uses_cooperative_matrix(module.data(), sizeof(module)))
        throw std::runtime_error("zero-length instruction accepted");
    module[0] = 0;
    if (spirv_uses_cooperative_matrix(module.data(), sizeof(module)))
        throw std::runtime_error("invalid module accepted");
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "usage: dav2_kernel_probe ORIGINAL_SPV TILED_SPV DEVICE "
                     "[16|64|attention|conv|implicit|norm|softmax|softmax-half|flash]\n";
        return 2;
    }
    try {
        check_shader_classifier();
        dav2::VulkanContext context(static_cast<std::uint32_t>(std::stoul(argv[3])));
        const bool conv = argc == 5 && std::string(argv[4]) == "conv";
        const bool norm = argc == 5 && std::string(argv[4]) == "norm";
        const bool soft = argc == 5 && (std::string(argv[4]) == "softmax" ||
                                        std::string(argv[4]) == "softmax-half");
        if (!conv && !norm && !soft && !context.compute_capabilities().cooperative_matrix_fp16) {
            std::cerr << "SKIP: GPU does not support the required FP16 cooperative matrix\n";
            return 77;
        }
        auto old_code = read_spirv(argv[1]), new_code = read_spirv(argv[2]);
        if (soft) {
            softmax(context, old_code, new_code, std::string(argv[4]) == "softmax-half");
            return 0;
        }
        if (argc == 5 && std::string(argv[4]) == "implicit") {
            convolution(context, old_code, new_code, true);
            return 0;
        }
        if (argc == 5 && std::string(argv[4]) == "flash") {
            flash_attention(context, new_code);
            return 0;
        }
        if (conv) {
            convolution(context, old_code, new_code);
            return 0;
        }
        if (norm) {
            layer_norm(context, old_code, new_code);
            return 0;
        }
        if (argc == 5 && std::string(argv[4]) == "attention") {
            attention_score(context, old_code, new_code);
            return 0;
        }
        const unsigned original_rows = argc == 5 ? static_cast<unsigned>(std::stoul(argv[4])) : 16;
        if (original_rows != 16 && original_rows != 64)
            throw std::runtime_error("invalid reference row tile");
        auto original = context.create_pipeline(old_code.data(), old_code.size() * 4, 4, 12);
        auto tiled = context.create_pipeline(new_code.data(), new_code.size() * 4, 4, 12);
        for (const auto dimensions :
             {std::vector<unsigned>{17, 28, 20}, {401, 384, 1152}, {1370, 1024, 4096}}) {
            struct Parameters {
                unsigned rows, ic, oc;
            } p{dimensions[0], dimensions[1], dimensions[2]};
            std::vector<std::uint16_t> input(std::size_t(p.rows) * p.ic),
                weight(std::size_t(p.oc) * p.ic);
            std::vector<float> bias(p.oc);
            for (std::size_t n = 0; n < input.size(); ++n)
                input[n] = (n % 5 == 0 ? 0x3400 : 0x3800) | (n % 3 == 0 ? 0x8000 : 0);
            for (std::size_t n = 0; n < weight.size(); ++n)
                weight[n] = (n % 7 == 0 ? 0x3000 : 0x3400) | (n % 11 == 0 ? 0x8000 : 0);
            for (unsigned n = 0; n < p.oc; ++n) bias[n] = float(n % 13) * 0.01f;
            auto gpu_input = context.create_device_buffer(input.size() * 2);
            auto gpu_weight = context.create_device_buffer(weight.size() * 2);
            auto gpu_bias = context.create_device_buffer(bias.size() * 4);
            auto output = context.create_device_buffer(std::uint64_t(p.rows) * p.oc * 4);
            context.upload(gpu_input, input.data(), input.size() * 2);
            context.upload(gpu_weight, weight.data(), weight.size() * 2);
            context.upload(gpu_bias, bias.data(), bias.size() * 4);
            const auto result = compare<float>(
                context, original, tiled, output, {&output, &gpu_input, &gpu_weight, &gpu_bias}, p,
                {(p.oc + 15) / 16, (p.rows + original_rows - 1) / original_rows, 1},
                {(p.oc + 31) / 32, (p.rows + 63) / 64, 1});
            const auto& reference = result.original;
            const auto& candidate = result.candidate;
            double error = 0, magnitude = 0, maximum = 0;
            for (std::size_t n = 0; n < reference.size(); ++n) {
                if (!std::isfinite(candidate[n]))
                    throw std::runtime_error("non-finite tiled output");
                const double delta = std::abs(double(reference[n]) - candidate[n]);
                error += delta;
                magnitude += std::abs(reference[n]);
                maximum = std::max(maximum, delta);
            }
            if (error / std::max(magnitude, 1e-12) > 1e-5 || maximum > 1e-3)
                throw std::runtime_error("tiled GEMM differs from original kernel");
            // Independent CPU checks include all small tail cases and sampled
            // positions in the larger transformer shapes.
            for (std::size_t n = 0; n < candidate.size(); n += candidate.size() < 1000 ? 1 : 7919) {
                const unsigned row = unsigned(n / p.oc), col = unsigned(n % p.oc);
                double expected = bias[col];
                for (unsigned k = 0; k < p.ic; ++k) {
                    const auto a = input[std::size_t(row) * p.ic + k],
                               b = weight[std::size_t(col) * p.ic + k];
                    const double af = (a & 0x7fff) == 0x3400 ? 0.25 : 0.5;
                    const double bf = (b & 0x7fff) == 0x3000 ? 0.125 : 0.25;
                    expected += (a & 0x8000 ? -af : af) * (b & 0x8000 ? -bf : bf);
                }
                if (std::abs(expected - candidate[n]) > 1e-4)
                    throw std::runtime_error("GEMM CPU reference mismatch");
            }
            std::cout << "GEMM_OK rows=" << p.rows << " ic=" << p.ic << " oc=" << p.oc
                      << " original_ms=" << result.original_ms
                      << " tiled_ms=" << result.candidate_ms
                      << " relative_l1=" << error / std::max(magnitude, 1e-12)
                      << " max_error=" << maximum << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
