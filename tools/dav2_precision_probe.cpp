#include "depth_anything_v2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

dav2_encoder parse_encoder(const std::string& value) {
    if (value == "vits") return DAV2_ENCODER_VITS;
    if (value == "vitb") return DAV2_ENCODER_VITB;
    if (value == "vitl") return DAV2_ENCODER_VITL;
    throw std::invalid_argument("encoder must be vits, vitb, or vitl");
}

std::uint32_t parse_precision(const std::string& value) {
    if (value == "fp32") return DAV2_CREATE_FORCE_FP32;
    if (value == "fp16") return DAV2_CREATE_FORCE_FP16;
    if (value == "int8") return DAV2_CREATE_FORCE_INT8;
    throw std::invalid_argument("precision must be fp32, fp16, or int8");
}

int positive(const char* value, const char* name) {
    const long parsed = std::strtol(value, nullptr, 10);
    if (parsed <= 0 || parsed > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string(name) + " must be positive");
    return static_cast<int>(parsed);
}

int nonnegative(const char* value, const char* name) {
    const long parsed = std::strtol(value, nullptr, 10);
    if (parsed < 0 || parsed > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string(name) + " must be non-negative");
    return static_cast<int>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL {vits|vitb|vitl} {fp32|fp16|int8}"
                     " DEVICE WIDTH HEIGHT RUNS\n";
        return 2;
    }
    try {
        const dav2_encoder encoder = parse_encoder(argv[2]);
        const std::uint32_t flags = parse_precision(argv[3]);
        const int device = nonnegative(argv[4], "device");
        const int width = positive(argv[5], "width");
        const int height = positive(argv[6], "height");
        const int runs = positive(argv[7], "runs");
        if (width % 14 != 0 || height % 14 != 0)
            throw std::invalid_argument("width and height must be multiples of 14");
        dav2_create_options options{
            sizeof(options), DAV2_ABI_VERSION, encoder, device, flags};
        dav2_context* context = nullptr;
        const dav2_status created = dav2_create(argv[1], &options, &context);
        if (created != DAV2_STATUS_OK)
            throw std::runtime_error(dav2_last_error());
        const std::size_t pixels =
            static_cast<std::size_t>(width) * height;
        std::vector<float> input(pixels * 3);
        for (std::size_t index = 0; index < input.size(); ++index)
            input[index] = std::sin(static_cast<float>(index) * 0.001f);
        std::vector<float> output(pixels);
        std::vector<double> milliseconds;
        milliseconds.reserve(static_cast<std::size_t>(runs));
        for (int run = 0; run < runs; ++run) {
            const auto start = std::chrono::steady_clock::now();
            const dav2_status status = dav2_infer_tensor_f32(
                context, input.data(), width, height,
                output.data(), output.size());
            if (status != DAV2_STATUS_OK) {
                const std::string inference_error = dav2_last_error();
                dav2_destroy(context);
                throw std::runtime_error(inference_error);
            }
            milliseconds.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        }
        dav2_destroy(context);
        if (!std::all_of(output.begin(), output.end(), [](float value) {
                return std::isfinite(value);
            }))
            throw std::runtime_error("inference produced a non-finite value");
        std::sort(milliseconds.begin(), milliseconds.end());
        const auto range = std::minmax_element(output.begin(), output.end());
        double mae = 0.0;
        double rmse = 0.0;
        double maximum_error = 0.0;
        double normalized_mae = 0.0;
        double normalized_rmse = 0.0;
        double correlation = 0.0;
        if ((flags & (DAV2_CREATE_FORCE_FP16 | DAV2_CREATE_FORCE_INT8)) != 0u) {
            options.flags = DAV2_CREATE_FORCE_FP32;
            dav2_context* reference_context = nullptr;
            const dav2_status reference_created =
                dav2_create(argv[1], &options, &reference_context);
            if (reference_created != DAV2_STATUS_OK)
                throw std::runtime_error(dav2_last_error());
            std::vector<float> reference(pixels);
            const dav2_status reference_status = dav2_infer_tensor_f32(
                reference_context, input.data(), width, height,
                reference.data(), reference.size());
            dav2_destroy(reference_context);
            if (reference_status != DAV2_STATUS_OK)
                throw std::runtime_error(dav2_last_error());
            const auto reference_range =
                std::minmax_element(reference.begin(), reference.end());
            const double output_span = std::max(
                1.0e-12,
                static_cast<double>(*range.second) - *range.first);
            const double reference_span = std::max(
                1.0e-12,
                static_cast<double>(*reference_range.second) -
                    *reference_range.first);
            double output_mean = 0.0;
            double reference_mean = 0.0;
            for (std::size_t index = 0; index < output.size(); ++index) {
                const double error = std::abs(
                    static_cast<double>(output[index]) - reference[index]);
                mae += error;
                rmse += error * error;
                maximum_error = std::max(maximum_error, error);
                const double normalized_output =
                    (output[index] - *range.first) / output_span;
                const double normalized_reference =
                    (reference[index] - *reference_range.first) /
                    reference_span;
                const double normalized_error =
                    std::abs(normalized_output - normalized_reference);
                normalized_mae += normalized_error;
                normalized_rmse += normalized_error * normalized_error;
                output_mean += normalized_output;
                reference_mean += normalized_reference;
            }
            mae /= static_cast<double>(output.size());
            rmse = std::sqrt(rmse / static_cast<double>(output.size()));
            normalized_mae /= static_cast<double>(output.size());
            normalized_rmse = std::sqrt(
                normalized_rmse / static_cast<double>(output.size()));
            output_mean /= static_cast<double>(output.size());
            reference_mean /= static_cast<double>(output.size());
            double covariance = 0.0;
            double output_variance = 0.0;
            double reference_variance = 0.0;
            for (std::size_t index = 0; index < output.size(); ++index) {
                const double normalized_output =
                    (output[index] - *range.first) / output_span;
                const double normalized_reference =
                    (reference[index] - *reference_range.first) /
                    reference_span;
                const double output_delta = normalized_output - output_mean;
                const double reference_delta =
                    normalized_reference - reference_mean;
                covariance += output_delta * reference_delta;
                output_variance += output_delta * output_delta;
                reference_variance += reference_delta * reference_delta;
            }
            correlation = covariance / std::sqrt(std::max(
                1.0e-24, output_variance * reference_variance));
        }
        std::cout << "precision=" << argv[3]
                  << " device=" << device
                  << " size=" << width << 'x' << height
                  << " median_ms=" << milliseconds[milliseconds.size() / 2]
                  << " minimum=" << *range.first
                  << " maximum=" << *range.second;
        if ((flags & (DAV2_CREATE_FORCE_FP16 | DAV2_CREATE_FORCE_INT8)) != 0u)
            std::cout << " mae_vs_fp32=" << mae
                      << " rmse_vs_fp32=" << rmse
                      << " max_error_vs_fp32=" << maximum_error
                      << " normalized_mae=" << normalized_mae
                      << " normalized_rmse=" << normalized_rmse
                      << " correlation=" << correlation;
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
