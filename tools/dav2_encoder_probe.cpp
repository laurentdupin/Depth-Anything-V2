#include "encoder.h"
#include "gpu_model.h"
#include "model.h"
#include "operators.h"
#include "vulkan.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

dav2_encoder parse_encoder(const std::string& text) {
    if (text == "vits") return DAV2_ENCODER_VITS;
    if (text == "vitb") return DAV2_ENCODER_VITB;
    if (text == "vitl") return DAV2_ENCODER_VITL;
    throw std::invalid_argument("encoder must be vits, vitb, or vitl");
}

std::vector<float> read_floats(
    const std::string& path,
    std::size_t count) {
    std::vector<float> result(count);
    std::ifstream input(path, std::ios::binary);
    input.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(result.size() * sizeof(float)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("invalid input tensor file");
    }
    return result;
}

void write_floats(
    const std::string& path,
    const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write feature file");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr
            << "usage: dav2_encoder_probe MODEL ENCODER WIDTH HEIGHT "
               "INPUT_F32 OUTPUT_PREFIX\n";
        return 2;
    }
    try {
        const dav2_encoder encoder = parse_encoder(argv[2]);
        const std::uint32_t width =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        const std::uint32_t height =
            static_cast<std::uint32_t>(std::stoul(argv[4]));
        std::vector<float> input =
            read_floats(argv[5], std::size_t(width) * height * 3);
        dav2::ModelFile model(argv[1], encoder);
        dav2::VulkanContext context(0);
        dav2::GpuModel weights(model, context);
        dav2::VulkanOperators operators(context);
        dav2::DinoEncoder implementation(
            encoder, context, weights, operators,
            inferbridge::native::Precision::fp32, false);
        auto gpu_input =
            context.create_device_buffer(input.size() * sizeof(float));
        context.upload(
            gpu_input, input.data(), input.size() * sizeof(float));
        const std::uint32_t embedding =
            encoder == DAV2_ENCODER_VITS ? 384 :
            encoder == DAV2_ENCODER_VITB ? 768 : 1024;
        const std::uint32_t token_count =
            (width / 14) * (height / 14) + 1;
        auto prepared = context.create_device_buffer(
            std::size_t(token_count) * embedding * sizeof(float));
        operators.prepare_tokens(
            prepared,
            gpu_input,
            weights.tensor("pretrained.patch_embed.proj.weight").buffer,
            weights.tensor("pretrained.patch_embed.proj.weight").half_buffer,
            weights.tensor("pretrained.patch_embed.proj.weight").int8_buffer,
            weights.tensor("pretrained.patch_embed.proj.weight").int8_scales,
            weights.tensor("pretrained.patch_embed.proj.bias").buffer,
            weights.tensor("pretrained.cls_token").buffer,
            weights.tensor("pretrained.pos_embed").buffer,
            width,
            height,
            embedding,
            inferbridge::native::Precision::fp32);
        std::vector<float> prepared_host(
            std::size_t(token_count) * embedding);
        context.download(
            prepared,
            prepared_host.data(),
            prepared_host.size() * sizeof(float));
        write_floats(
            std::string(argv[6]) + ".tokens.f32",
            prepared_host);
        dav2::EncoderOutput encoded =
            implementation.forward(gpu_input, width, height);
        const std::size_t elements =
            std::size_t(encoded.tokens) * encoded.embedding;
        for (std::size_t index = 0;
             index < encoded.features.size();
             ++index) {
            std::vector<float> feature(elements);
            context.download(
                encoded.features[index],
                feature.data(),
                feature.size() * sizeof(float));
            write_floats(
                std::string(argv[6]) + "." + std::to_string(index) + ".f32",
                feature);
        }
        std::cout << encoded.tokens << ' ' << encoded.embedding << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
