#include "depth_anything_v2.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {

bool wait_for_job(dav2_gpu_job* job) {
    for (int attempt = 0; attempt < 30000; ++attempt) {
        dav2_gpu_job_status status{};
        status.struct_size = sizeof(status);
        const dav2_status result = dav2_gpu_job_poll(job, &status);
        if (result != DAV2_STATUS_OK) {
            std::cerr << "poll: " << dav2_last_error() << '\n';
            return false;
        }
        if (status.state == DAV2_GPU_JOB_COMPLETE) return true;
        if (status.state == DAV2_GPU_JOB_FAILED ||
            status.state == DAV2_GPU_JOB_CANCELLED) {
            std::cerr << "Metal job entered terminal state "
                      << status.state << '\n';
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::cerr << "Metal job timed out\n";
    return false;
}

}  // namespace

int main() {
    const char* model_path = std::getenv("DAV2_VITS_MODEL");
    if (model_path == nullptr || *model_path == '\0') {
        std::cout << "DAV2_VITS_MODEL is not set; skipping\n";
        return 77;
    }
    @autoreleasepool {
        constexpr uint32_t width = 336u;
        constexpr uint32_t height = 210u;
        constexpr uint32_t input_size = 140u;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::cerr << "Metal device is unavailable\n";
            return 1;
        }
        MTLTextureDescriptor* input_description =
            [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                width:width height:height mipmapped:NO];
        input_description.storageMode = MTLStorageModeShared;
        input_description.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> input =
            [device newTextureWithDescriptor:input_description];
        MTLTextureDescriptor* output_description =
            [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                width:width height:height mipmapped:NO];
        output_description.storageMode = MTLStorageModePrivate;
        output_description.usage = MTLTextureUsageShaderRead |
            MTLTextureUsageShaderWrite;
        id<MTLTexture> output =
            [device newTextureWithDescriptor:output_description];
        id<MTLSharedEvent> event = [device newSharedEvent];
        if (input == nil || output == nil || event == nil) {
            std::cerr << "Metal test resources could not be allocated\n";
            return 2;
        }
        std::vector<uint8_t> pixels(
            static_cast<size_t>(width) * height * 4u);
        for (uint32_t y = 0u; y < height; ++y) {
            for (uint32_t x = 0u; x < width; ++x) {
                const size_t offset =
                    (static_cast<size_t>(y) * width + x) * 4u;
                pixels[offset] = static_cast<uint8_t>((x + y) & 255u);
                pixels[offset + 1u] = static_cast<uint8_t>(
                    (x * 3u + y * 5u) & 255u);
                pixels[offset + 2u] = static_cast<uint8_t>(
                    (x * 7u + y * 2u) & 255u);
                pixels[offset + 3u] = 255u;
            }
        }
        [input replaceRegion:MTLRegionMake2D(0u, 0u, width, height)
                 mipmapLevel:0u
                   withBytes:pixels.data()
                 bytesPerRow:width * 4u];

        const dav2_create_options options{
            sizeof(options), DAV2_ABI_VERSION, DAV2_ENCODER_VITS, 0,
            DAV2_CREATE_FORCE_METAL};
        dav2_context* context = nullptr;
        dav2_status status = dav2_create(model_path, &options, &context);
        if (status != DAV2_STATUS_OK) {
            std::cerr << "create: " << dav2_last_error() << '\n';
            return 3;
        }
        dav2_gpu_capabilities capabilities{};
        capabilities.struct_size = sizeof(capabilities);
        status = dav2_get_gpu_capabilities(context, &capabilities);
        const uint64_t required =
            DAV2_GPU_CAP_METAL_TEXTURE_INPUT |
            DAV2_GPU_CAP_METAL_SHARED_EVENT_WAIT |
            DAV2_GPU_CAP_METAL_TEXTURE_OUTPUT |
            DAV2_GPU_CAP_METAL_SHARED_EVENT_SIGNAL |
            DAV2_GPU_CAP_NO_HOST_PIXEL_STAGING |
            DAV2_GPU_CAP_NO_HOST_DEPTH_STAGING;
        if (status != DAV2_STATUS_OK ||
            (capabilities.flags & required) != required ||
            capabilities.adapter_luid != device.registryID) {
            std::cerr << "Metal capability contract is incomplete\n";
            dav2_destroy(context);
            return 4;
        }
        std::vector<double> samples;
        for (uint64_t iteration = 1u; iteration <= 5u; ++iteration) {
            const dav2_metal_texture_binding_request request{
                sizeof(request), DAV2_ABI_VERSION,
                reinterpret_cast<uintptr_t>(input),
                width, height, DAV2_GPU_PIXEL_BGRA8,
                static_cast<int32_t>(input_size),
                0u, 0u,
                reinterpret_cast<uintptr_t>(output),
                width, height,
                reinterpret_cast<uintptr_t>(event), iteration,
                iteration, iteration * 100u};
            dav2_gpu_job* job = nullptr;
            const auto start = std::chrono::steady_clock::now();
            status = dav2_submit_metal_texture_binding(
                context, &request, &job);
            if (status != DAV2_STATUS_OK || job == nullptr) {
                std::cerr << "submit: " << dav2_last_error() << '\n';
                dav2_destroy(context);
                return 5;
            }
            const bool completed = wait_for_job(job);
            const double elapsed =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
            dav2_gpu_job_release(job);
            if (!completed) {
                dav2_destroy(context);
                return 6;
            }
            if (iteration > 1u) samples.push_back(elapsed);
        }

        constexpr NSUInteger row_bytes = 1536u;
        const NSUInteger buffer_size = row_bytes * height;
        id<MTLBuffer> readback = [device
            newBufferWithLength:buffer_size
            options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        [blit copyFromTexture:output sourceSlice:0u sourceLevel:0u
                sourceOrigin:MTLOriginMake(0u, 0u, 0u)
                  sourceSize:MTLSizeMake(width, height, 1u)
                    toBuffer:readback destinationOffset:0u
      destinationBytesPerRow:row_bytes
    destinationBytesPerImage:buffer_size];
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        float minimum = 1.0f;
        float maximum = 0.0f;
        for (uint32_t y = 0u; y < height; ++y) {
            const auto* row = reinterpret_cast<const float*>(
                static_cast<const uint8_t*>(readback.contents) +
                static_cast<size_t>(y) * row_bytes);
            for (uint32_t x = 0u; x < width; ++x) {
                if (!std::isfinite(row[x]) || row[x] < -1.0e-5f ||
                    row[x] > 1.00001f) {
                    std::cerr << "invalid Metal depth value " << row[x] << '\n';
                    dav2_destroy(context);
                    return 7;
                }
                minimum = std::min(minimum, row[x]);
                maximum = std::max(maximum, row[x]);
            }
        }
        if (maximum - minimum < 0.01f) {
            std::cerr << "Metal depth output has no useful range\n";
            dav2_destroy(context);
            return 8;
        }
        dav2_transfer_counters counters{};
        counters.struct_size = sizeof(counters);
        status = dav2_get_transfer_counters(context, &counters);
        if (status != DAV2_STATUS_OK || counters.tensor_upload_bytes != 0u ||
            counters.tensor_download_bytes != 0u) {
            std::cerr << "Metal texture path performed host tensor staging\n";
            dav2_destroy(context);
            return 9;
        }
        std::sort(samples.begin(), samples.end());
        std::cout << "Metal texture full graph median: "
                  << samples[samples.size() / 2u] << " ms, range "
                  << minimum << " .. " << maximum << '\n';
        dav2_destroy(context);
        [readback release];
        [queue release];
        [event release];
        [output release];
        [input release];
        [device release];
    }
    return 0;
}
