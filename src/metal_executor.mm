#include "executor.h"
#include "model.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dav2 {
namespace {

MPSShape* shape(std::initializer_list<NSInteger> dimensions) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:dimensions.size()];
    for (NSInteger value : dimensions) [result addObject:@(value)];
    return result;
}

MPSShape* shape(const TensorView& tensor) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:tensor.rank];
    for (std::uint32_t index = 0; index < tensor.rank; ++index) {
        [result addObject:@(static_cast<unsigned long long>(
            tensor.dimensions[index]))];
    }
    return result;
}

NSString* name(const std::string& value) {
    return [NSString stringWithUTF8String:value.c_str()];
}

float cubic_convolution1(float x) {
    constexpr float a = -0.75f;
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
}

float cubic_convolution2(float x) {
    constexpr float a = -0.75f;
    return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
}

std::array<float, 4> cubic_coefficients(float t) {
    return {
        cubic_convolution2(t + 1.0f),
        cubic_convolution1(t),
        cubic_convolution1(1.0f - t),
        cubic_convolution2(2.0f - t),
    };
}

std::vector<float> position_embedding(
    const TensorView& source, int patch_height, int patch_width,
    int embedding) {
    if (source.rank != 3u || source.dimensions[0] != 1u ||
        source.dimensions[1] != 1370u ||
        source.dimensions[2] != static_cast<std::uint64_t>(embedding)) {
        throw std::runtime_error("invalid DINO position embedding");
    }
    const int patches = patch_height * patch_width;
    std::vector<float> result(
        static_cast<std::size_t>(patches + 1) * embedding);
    std::copy_n(source.data, embedding, result.data());

    const float scale_y = 37.0f / (static_cast<float>(patch_height) + 0.1f);
    const float scale_x = 37.0f / (static_cast<float>(patch_width) + 0.1f);
    for (int y = 0; y < patch_height; ++y) {
        const float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int base_y = static_cast<int>(std::floor(source_y));
        const auto cy = cubic_coefficients(source_y - base_y);
        for (int x = 0; x < patch_width; ++x) {
            const float source_x =
                (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int base_x = static_cast<int>(std::floor(source_x));
            const auto cx = cubic_coefficients(source_x - base_x);
            float* destination = result.data() +
                static_cast<std::size_t>(y * patch_width + x + 1) * embedding;
            for (int channel = 0; channel < embedding; ++channel) {
                float value = 0.0f;
                for (int ky = 0; ky < 4; ++ky) {
                    const int sy = std::clamp(base_y - 1 + ky, 0, 36);
                    float row = 0.0f;
                    for (int kx = 0; kx < 4; ++kx) {
                        const int sx = std::clamp(base_x - 1 + kx, 0, 36);
                        const std::size_t offset =
                            (static_cast<std::size_t>(1 + sy * 37 + sx) *
                             embedding) + channel;
                        row += source.data[offset] * cx[kx];
                    }
                    value += row * cy[ky];
                }
                destination[channel] = value;
            }
        }
    }
    return result;
}

struct EncoderConfiguration {
    int embedding = 0;
    int heads = 0;
    int blocks = 0;
    int features = 0;
    std::array<int, 4> captures{};
    std::array<int, 4> project_channels{};
};

EncoderConfiguration configuration(dav2_encoder encoder) {
    switch (encoder) {
        case DAV2_ENCODER_VITS:
            return {384, 6, 12, 64, {2, 5, 8, 11}, {48, 96, 192, 384}};
        case DAV2_ENCODER_VITB:
            return {768, 12, 12, 128, {2, 5, 8, 11}, {96, 192, 384, 768}};
        case DAV2_ENCODER_VITL:
            return {1024, 16, 24, 256, {4, 11, 17, 23},
                    {256, 512, 1024, 1024}};
        default:
            throw std::invalid_argument("unsupported DINOv2 encoder");
    }
}

class MetalGraphBuilder {
public:
    MetalGraphBuilder(
        const ModelFile& model, dav2_encoder encoder,
        int width, int height, int output_width, int output_height)
        : model_(model), config_(configuration(encoder)),
          width_(width), height_(height),
          patch_width_(width / 14), patch_height_(height / 14),
          tokens_(patch_width_ * patch_height_ + 1),
          output_width_(output_width), output_height_(output_height),
          graph_([MPSGraph new]) {}

    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* input() const { return input_; }
    MPSGraphTensor* output() const { return output_; }

    void build(float metric_max_depth) {
        input_ = [graph_ placeholderWithShape:shape({1, 3, height_, width_})
                                         dataType:MPSDataTypeFloat32
                                             name:@"normalized_rgb_chw"];
        MPSGraphTensor* current = conv(
            input_, "pretrained.patch_embed.proj", 14, 0);
        current = [graph_ reshapeTensor:current
                              withShape:shape({1, config_.embedding,
                                               patch_height_ * patch_width_})
                                   name:nil];
        current = [graph_ transposeTensor:current
                                dimension:1
                            withDimension:2
                                     name:nil];
        MPSGraphTensor* class_token = constant("pretrained.cls_token");
        current = [graph_ concatTensors:@[class_token, current]
                               dimension:1
                                    name:nil];
        const std::vector<float> positions = position_embedding(
            model_.tensor("pretrained.pos_embed"), patch_height_,
            patch_width_, config_.embedding);
        MPSGraphTensor* position = owned_constant(
            positions, shape({1, tokens_, config_.embedding}));
        current = add(current, position);

        std::array<MPSGraphTensor*, 4> captured{};
        int capture_index = 0;
        for (int block = 0; block < config_.blocks; ++block) {
            const std::string prefix =
                "pretrained.blocks." + std::to_string(block);
            MPSGraphTensor* normalized = layer_norm(
                current, prefix + ".norm1", 1.0e-6f);
            MPSGraphTensor* qkv = linear(
                normalized, prefix + ".attn.qkv");
            qkv = [graph_ reshapeTensor:qkv
                              withShape:shape({1, tokens_, 3, config_.heads, 64})
                                   name:nil];
            qkv = [graph_ transposeTensor:qkv
                              permutation:@[@2, @0, @3, @1, @4]
                                     name:nil];
            MPSGraphTensor* query = slice(qkv, 0, 0, 1);
            MPSGraphTensor* key = slice(qkv, 0, 1, 1);
            MPSGraphTensor* value = slice(qkv, 0, 2, 1);
            query = [graph_ reshapeTensor:query
                               withShape:shape({1, config_.heads, tokens_, 64})
                                    name:nil];
            key = [graph_ reshapeTensor:key
                             withShape:shape({1, config_.heads, tokens_, 64})
                                  name:nil];
            value = [graph_ reshapeTensor:value
                               withShape:shape({1, config_.heads, tokens_, 64})
                                    name:nil];
            query = multiply(query, scalar(0.125f));
            key = [graph_ transposeTensor:key
                                dimension:2
                            withDimension:3
                                     name:nil];
            MPSGraphTensor* scores = [graph_
                matrixMultiplicationWithPrimaryTensor:query
                secondaryTensor:key
                name:nil];
            scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
            MPSGraphTensor* attention = [graph_
                matrixMultiplicationWithPrimaryTensor:scores
                secondaryTensor:value
                name:nil];
            attention = [graph_ transposeTensor:attention
                                      dimension:1
                                  withDimension:2
                                           name:nil];
            attention = [graph_ reshapeTensor:attention
                                    withShape:shape({1, tokens_, config_.embedding})
                                         name:nil];
            MPSGraphTensor* projected = linear(
                attention, prefix + ".attn.proj");
            projected = multiply(
                projected, constant(prefix + ".ls1.gamma"));
            current = add(current, projected);

            normalized = layer_norm(
                current, prefix + ".norm2", 1.0e-6f);
            MPSGraphTensor* hidden = linear(
                normalized, prefix + ".mlp.fc1");
            hidden = gelu(hidden);
            hidden = linear(hidden, prefix + ".mlp.fc2");
            hidden = multiply(hidden, constant(prefix + ".ls2.gamma"));
            current = add(current, hidden);

            if (capture_index < 4 &&
                block == config_.captures[capture_index]) {
                captured[capture_index] = layer_norm(
                    current, "pretrained.norm", 1.0e-6f);
                ++capture_index;
            }
        }
        if (capture_index != 4) {
            throw std::runtime_error("Metal encoder did not capture four features");
        }
        output_ = build_dpt(captured, metric_max_depth);
    }

private:
    MPSGraphTensor* constant(const std::string& tensor_name) {
        const TensorView& tensor = model_.tensor(tensor_name);
        NSData* data = [NSData dataWithBytesNoCopy:
            const_cast<float*>(tensor.data)
            length:static_cast<NSUInteger>(tensor.elements * sizeof(float))
            freeWhenDone:NO];
        return [graph_ constantWithData:data
                                  shape:shape(tensor)
                               dataType:MPSDataTypeFloat32];
    }

    MPSGraphTensor* owned_constant(
        const std::vector<float>& values, MPSShape* dimensions) {
        NSData* data = [NSData dataWithBytes:values.data()
                                      length:values.size() * sizeof(float)];
        return [graph_ constantWithData:data
                                  shape:dimensions
                               dataType:MPSDataTypeFloat32];
    }

    MPSGraphTensor* scalar(float value) {
        return [graph_ constantWithScalar:value dataType:MPSDataTypeFloat32];
    }

    MPSGraphTensor* add(MPSGraphTensor* left, MPSGraphTensor* right) {
        return [graph_ additionWithPrimaryTensor:left
                                secondaryTensor:right
                                           name:nil];
    }

    MPSGraphTensor* multiply(MPSGraphTensor* left, MPSGraphTensor* right) {
        return [graph_ multiplicationWithPrimaryTensor:left
                                      secondaryTensor:right
                                                 name:nil];
    }

    MPSGraphTensor* slice(
        MPSGraphTensor* tensor, NSUInteger dimension,
        NSInteger start, NSInteger length) {
        return [graph_ sliceTensor:tensor dimension:dimension
                             start:start length:length name:nil];
    }

    MPSGraphTensor* linear(
        MPSGraphTensor* tensor, const std::string& prefix) {
        MPSGraphTensor* weight = constant(prefix + ".weight");
        weight = [graph_ transposeTensor:weight
                               dimension:0
                           withDimension:1
                                    name:nil];
        MPSGraphTensor* result = [graph_
            matrixMultiplicationWithPrimaryTensor:tensor
            secondaryTensor:weight
            name:name(prefix)];
        if (model_.contains(prefix + ".bias")) {
            result = add(result, constant(prefix + ".bias"));
        }
        return result;
    }

    MPSGraphTensor* layer_norm(
        MPSGraphTensor* tensor, const std::string& prefix, float epsilon) {
        NSArray<NSNumber*>* axes = @[@(-1)];
        MPSGraphTensor* mean = [graph_ meanOfTensor:tensor axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:tensor
                                                meanTensor:mean
                                                      axes:axes
                                                      name:nil];
        return [graph_ normalizationWithTensor:tensor
                                    meanTensor:mean
                                varianceTensor:variance
                                   gammaTensor:constant(prefix + ".weight")
                                    betaTensor:constant(prefix + ".bias")
                                       epsilon:epsilon
                                          name:name(prefix)];
    }

    MPSGraphTensor* gelu(MPSGraphTensor* tensor) {
        MPSGraphTensor* scaled = multiply(
            tensor, scalar(static_cast<float>(M_SQRT1_2)));
        MPSGraphTensor* error = [graph_ erfWithTensor:scaled name:nil];
        return multiply(
            multiply(tensor, scalar(0.5f)), add(error, scalar(1.0f)));
    }

    MPSGraphConvolution2DOpDescriptor* convolution_descriptor(
        int stride, int padding) {
        return [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride
            strideInY:stride
            dilationRateInX:1
            dilationRateInY:1
            groups:1
            paddingLeft:padding
            paddingRight:padding
            paddingTop:padding
            paddingBottom:padding
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    }

    MPSGraphTensor* conv(
        MPSGraphTensor* tensor, const std::string& prefix,
        int stride, int padding, bool bias = true) {
        MPSGraphTensor* result = [graph_
            convolution2DWithSourceTensor:tensor
            weightsTensor:constant(prefix + ".weight")
            descriptor:convolution_descriptor(stride, padding)
            name:name(prefix)];
        if (bias && model_.contains(prefix + ".bias")) {
            const TensorView& bias_tensor = model_.tensor(prefix + ".bias");
            MPSGraphTensor* bias_value = [graph_ reshapeTensor:
                constant(prefix + ".bias")
                withShape:shape({1, static_cast<NSInteger>(bias_tensor.elements), 1, 1})
                name:nil];
            result = add(result, bias_value);
        }
        return result;
    }

    MPSGraphTensor* conv_transpose(
        MPSGraphTensor* tensor, const std::string& prefix,
        int channels, int source_height, int source_width, int stride) {
        MPSGraphTensor* result = [graph_
            convolutionTranspose2DWithSourceTensor:tensor
            weightsTensor:constant(prefix + ".weight")
            outputShape:shape({1, channels, source_height * stride,
                               source_width * stride})
            descriptor:convolution_descriptor(stride, 0)
            name:name(prefix)];
        MPSGraphTensor* bias = [graph_ reshapeTensor:
            constant(prefix + ".bias")
            withShape:shape({1, channels, 1, 1})
            name:nil];
        return add(result, bias);
    }

    MPSGraphTensor* resize(
        MPSGraphTensor* tensor, int height, int width,
        MPSGraphResizeMode mode, bool align_corners) {
        return [graph_ resizeTensor:tensor
                              size:shape({height, width})
                              mode:mode
                      centerResult:align_corners ? NO : YES
                      alignCorners:align_corners ? YES : NO
                            layout:MPSGraphTensorNamedDataLayoutNCHW
                              name:nil];
    }

    MPSGraphTensor* residual_unit(
        MPSGraphTensor* tensor, const std::string& prefix) {
        MPSGraphTensor* result = [graph_ reLUWithTensor:tensor name:nil];
        result = conv(result, prefix + ".conv1", 1, 1);
        result = [graph_ reLUWithTensor:result name:nil];
        result = conv(result, prefix + ".conv2", 1, 1);
        return add(tensor, result);
    }

    MPSGraphTensor* fusion(
        MPSGraphTensor* path, MPSGraphTensor* skip,
        const std::string& prefix, int target_height, int target_width) {
        if (skip != nil) {
            path = add(path, residual_unit(
                skip, prefix + ".resConfUnit1"));
        }
        path = residual_unit(path, prefix + ".resConfUnit2");
        path = resize(
            path, target_height, target_width,
            MPSGraphResizeBilinear, true);
        return conv(path, prefix + ".out_conv", 1, 0);
    }

    MPSGraphTensor* build_dpt(
        const std::array<MPSGraphTensor*, 4>& features,
        float metric_max_depth) {
        std::array<MPSGraphTensor*, 4> layers{};
        std::array<int, 4> layer_heights{
            patch_height_ * 4, patch_height_ * 2,
            patch_height_, (patch_height_ + 1) / 2};
        std::array<int, 4> layer_widths{
            patch_width_ * 4, patch_width_ * 2,
            patch_width_, (patch_width_ + 1) / 2};
        for (int index = 0; index < 4; ++index) {
            MPSGraphTensor* patches = slice(
                features[index], 1, 1, patch_width_ * patch_height_);
            patches = [graph_ transposeTensor:patches
                                    dimension:1
                                withDimension:2
                                         name:nil];
            patches = [graph_ reshapeTensor:patches
                                  withShape:shape({1, config_.embedding,
                                                   patch_height_, patch_width_})
                                       name:nil];
            const std::string project =
                "depth_head.projects." + std::to_string(index);
            patches = conv(patches, project, 1, 0);
            if (index < 2) {
                const int scale = index == 0 ? 4 : 2;
                patches = conv_transpose(
                    patches,
                    "depth_head.resize_layers." + std::to_string(index),
                    config_.project_channels[index],
                    patch_height_, patch_width_, scale);
            } else if (index == 3) {
                patches = conv(
                    patches, "depth_head.resize_layers.3", 2, 1);
            }
            layers[index] = conv(
                patches,
                "depth_head.scratch.layer" + std::to_string(index + 1) +
                    "_rn",
                1, 1, false);
        }

        MPSGraphTensor* path = fusion(
            layers[3], nil, "depth_head.scratch.refinenet4",
            layer_heights[2], layer_widths[2]);
        path = fusion(
            path, layers[2], "depth_head.scratch.refinenet3",
            layer_heights[1], layer_widths[1]);
        path = fusion(
            path, layers[1], "depth_head.scratch.refinenet2",
            layer_heights[0], layer_widths[0]);
        path = fusion(
            path, layers[0], "depth_head.scratch.refinenet1",
            layer_heights[0] * 2, layer_widths[0] * 2);
        path = conv(
            path, "depth_head.scratch.output_conv1", 1, 1);
        path = resize(
            path, height_, width_, MPSGraphResizeBilinear, true);
        path = conv(
            path, "depth_head.scratch.output_conv2.0", 1, 1);
        path = [graph_ reLUWithTensor:path name:nil];
        path = conv(
            path, "depth_head.scratch.output_conv2.2", 1, 0);
        if (metric_max_depth > 0.0f) {
            path = [graph_ sigmoidWithTensor:path name:nil];
            path = multiply(path, scalar(metric_max_depth));
        } else {
            path = [graph_ reLUWithTensor:path name:nil];
        }
        if (output_height_ != height_ || output_width_ != width_) {
            path = resize(
                path, output_height_, output_width_,
                MPSGraphResizeBilinear, true);
        }
        return path;
    }

    const ModelFile& model_;
    EncoderConfiguration config_;
    int width_;
    int height_;
    int patch_width_;
    int patch_height_;
    int tokens_;
    int output_width_;
    int output_height_;
    MPSGraph* graph_;
    MPSGraphTensor* input_ = nil;
    MPSGraphTensor* output_ = nil;
};

struct PlanKey {
    int width;
    int height;
    int output_width;
    int output_height;

    bool operator==(const PlanKey& other) const {
        return width == other.width && height == other.height &&
            output_width == other.output_width &&
            output_height == other.output_height;
    }
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey& value) const {
        std::size_t result = static_cast<std::size_t>(value.width);
        result = result * 1315423911u + static_cast<std::size_t>(value.height);
        result = result * 1315423911u +
            static_cast<std::size_t>(value.output_width);
        return result * 1315423911u +
            static_cast<std::size_t>(value.output_height);
    }
};

struct MetalPlan {
    MPSGraph* graph = nil;
    MPSGraphTensor* input = nil;
    MPSGraphTensor* output = nil;
    MPSGraphExecutable* executable = nil;
    int width = 0;
    int height = 0;
    int output_width = 0;
    int output_height = 0;
};

class MetalExecutor final : public Executor {
public:
    MetalExecutor(
        const std::string& model_path, dav2_encoder encoder,
        std::uint32_t flags)
        : model_(model_path, encoder), encoder_(encoder),
          metric_max_depth_(model_.derivation().metric_max_depth) {
        if ((flags & DAV2_CREATE_FORCE_INT8) != 0u) {
            throw std::invalid_argument(
                "the Metal executor does not support INT8 yet");
        }
        if ((flags & DAV2_CREATE_FORCE_FP16) != 0u) {
            throw std::invalid_argument(
                "the Metal executor does not support forced FP16 yet");
        }
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) {
            throw std::runtime_error("Metal is unavailable on this Mac");
        }
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (queue_ == nil || graph_device_ == nil) {
            throw std::runtime_error("could not initialize the Metal executor");
        }
    }

    void infer(
        const float* input, int width, int height, float* depth) override {
        infer_resized(input, width, height, depth, width, height);
    }

    void infer_resized(
        const float* input, int width, int height, float* depth,
        int output_width, int output_height) override {
        if (input == nullptr || depth == nullptr || width <= 0 || height <= 0 ||
            output_width <= 0 || output_height <= 0 ||
            width % 14 != 0 || height % 14 != 0) {
            throw std::invalid_argument("invalid Metal inference dimensions");
        }
        std::lock_guard<std::mutex> guard(mutex_);
        @autoreleasepool {
            MetalPlan& plan = get_plan(
                width, height, output_width, output_height);
            const NSUInteger input_bytes = static_cast<NSUInteger>(
                std::uint64_t(width) * height * 3u * sizeof(float));
            id<MTLBuffer> input_buffer = [device_
                newBufferWithBytes:input
                length:input_bytes
                options:MTLResourceStorageModeShared];
            if (input_buffer == nil) {
                throw std::bad_alloc();
            }
            MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:input_buffer
                shape:shape({1, 3, height, width})
                dataType:MPSDataTypeFloat32];
            MPSGraphExecutableExecutionDescriptor* execution =
                [MPSGraphExecutableExecutionDescriptor new];
            execution.waitUntilCompleted = YES;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runWithMTLCommandQueue:queue_
                inputsArray:@[input_data]
                resultsArray:nil
                executionDescriptor:execution];
            if (results.count != 1u) {
                throw std::runtime_error("Metal graph returned no depth tensor");
            }
            [results[0].mpsndarray readBytes:depth strideBytes:nil];
            upload_bytes_.fetch_add(input_bytes, std::memory_order_relaxed);
            download_bytes_.fetch_add(
                std::uint64_t(output_width) * output_height * sizeof(float),
                std::memory_order_relaxed);
        }
    }

    GpuCapabilities gpu_capabilities() const override { return {}; }

    std::unique_ptr<GpuJob> submit_gpu(
        const GpuSubmitRequest&) override {
        throw std::runtime_error(
            "Metal external-buffer submission is not implemented yet");
    }

    std::unique_ptr<GpuJob> submit_gpu_texture(
        const GpuTextureSubmitRequest&) override {
        throw std::runtime_error(
            "Metal external-texture submission is not implemented yet");
    }

    void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const override {
        upload_bytes = upload_bytes_.load(std::memory_order_relaxed);
        download_bytes = download_bytes_.load(std::memory_order_relaxed);
    }

    dav2_encoder encoder() const override { return encoder_; }
    float metric_max_depth() const override { return metric_max_depth_; }

private:
    MetalPlan& get_plan(
        int width, int height, int output_width, int output_height) {
        const PlanKey key{width, height, output_width, output_height};
        auto existing = plans_.find(key);
        if (existing != plans_.end()) return existing->second;

        MetalGraphBuilder builder(
            model_, encoder_, width, height, output_width, output_height);
        builder.build(metric_max_depth_);
        MPSGraphShapedType* input_type = [[MPSGraphShapedType alloc]
            initWithShape:shape({1, 3, height, width})
            dataType:MPSDataTypeFloat32];
        MPSGraphCompilationDescriptor* compilation =
            [MPSGraphCompilationDescriptor new];
        compilation.optimizationLevel = MPSGraphOptimizationLevel1;
        compilation.waitForCompilationCompletion = YES;
        MPSGraphExecutable* executable = [builder.graph()
            compileWithDevice:graph_device_
            feeds:@{builder.input(): input_type}
            targetTensors:@[builder.output()]
            targetOperations:nil
            compilationDescriptor:compilation];
        if (executable == nil) {
            throw std::runtime_error("failed to compile the Metal graph");
        }
        executable.options = MPSGraphOptionsSynchronizeResults;
        MetalPlan plan;
        plan.graph = builder.graph();
        plan.input = builder.input();
        plan.output = builder.output();
        plan.executable = executable;
        plan.width = width;
        plan.height = height;
        plan.output_width = output_width;
        plan.output_height = output_height;
        return plans_.emplace(key, std::move(plan)).first->second;
    }

    ModelFile model_;
    dav2_encoder encoder_;
    float metric_max_depth_ = 0.0f;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSGraphDevice* graph_device_ = nil;
    std::unordered_map<PlanKey, MetalPlan, PlanKeyHash> plans_;
    std::mutex mutex_;
    std::atomic<std::uint64_t> upload_bytes_{0u};
    std::atomic<std::uint64_t> download_bytes_{0u};
};

}  // namespace

std::unique_ptr<Executor> create_metal_executor(
    const std::string& model_path,
    dav2_encoder encoder,
    std::uint32_t flags) {
    return std::make_unique<MetalExecutor>(model_path, encoder, flags);
}

}  // namespace dav2
