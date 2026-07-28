#include "depth_anything_v2.h"

#include "executor.h"
#include "image.h"

#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct dav2_context {
    std::unique_ptr<dav2::Executor> executor;
    dav2::ImageScratch image_scratch;
    std::vector<float> image_input;
};

namespace {

thread_local std::string last_error;

dav2_status fail(dav2_status status, const char* message) {
    last_error = message ? message : "";
    return status;
}

bool supported_encoder(dav2_encoder encoder) {
    return encoder == DAV2_ENCODER_VITS ||
        encoder == DAV2_ENCODER_VITB ||
        encoder == DAV2_ENCODER_VITL;
}

template <typename Function>
dav2_status protect(Function&& function) {
    try {
        function();
        last_error.clear();
        return DAV2_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(DAV2_STATUS_OUT_OF_MEMORY, "out of memory");
    } catch (const std::invalid_argument& error) {
        return fail(DAV2_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return fail(DAV2_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(DAV2_STATUS_INTERNAL_ERROR, "unknown internal error");
    }
}

}  // namespace

#if !defined(DAV2_WITH_VULKAN)
namespace dav2 {
std::unique_ptr<Executor> create_executor(
    const std::string&,
    dav2_encoder,
    int) {
    throw std::runtime_error(
        "this DLL was built without Vulkan");
}
}  // namespace dav2
#endif

extern "C" {

uint32_t DAV2_CALL dav2_abi_version(void) {
    return DAV2_ABI_VERSION;
}

const char* DAV2_CALL dav2_version_string(void) {
    return "0.1.0";
}

const char* DAV2_CALL dav2_status_string(dav2_status status) {
    switch (status) {
        case DAV2_STATUS_OK: return "ok";
        case DAV2_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case DAV2_STATUS_MODEL_IO: return "model I/O error";
        case DAV2_STATUS_MODEL_FORMAT: return "invalid model format";
        case DAV2_STATUS_VULKAN_UNAVAILABLE: return "Vulkan unavailable";
        case DAV2_STATUS_OUT_OF_MEMORY: return "out of memory";
        case DAV2_STATUS_INFERENCE_FAILED: return "inference failed";
        case DAV2_STATUS_BUFFER_TOO_SMALL: return "buffer too small";
        case DAV2_STATUS_UNSUPPORTED: return "unsupported";
        case DAV2_STATUS_INTERNAL_ERROR: return "internal error";
        default: return "unknown status";
    }
}

const char* DAV2_CALL dav2_last_error(void) {
    return last_error.c_str();
}

dav2_status DAV2_CALL dav2_get_network_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    dav2_image_shape* network_shape) {
    if (network_shape == nullptr) {
        return fail(DAV2_STATUS_INVALID_ARGUMENT, "network_shape is null");
    }
    return protect([&] {
        const dav2::ImageShape shape =
            dav2::network_shape(image_width, image_height, input_size);
        network_shape->width = shape.width;
        network_shape->height = shape.height;
    });
}

dav2_status DAV2_CALL dav2_create(
    const char* model_path_utf8,
    const dav2_create_options* options,
    dav2_context** context) {
    if (context == nullptr) {
        return fail(DAV2_STATUS_INVALID_ARGUMENT, "context is null");
    }
    *context = nullptr;
    if (model_path_utf8 == nullptr || model_path_utf8[0] == '\0') {
        return fail(DAV2_STATUS_INVALID_ARGUMENT, "model_path_utf8 is empty");
    }
    if (options == nullptr || options->struct_size < sizeof(dav2_create_options)) {
        return fail(DAV2_STATUS_INVALID_ARGUMENT, "invalid create options");
    }
    if (options->abi_version != DAV2_ABI_VERSION) {
        return fail(DAV2_STATUS_INVALID_ARGUMENT, "ABI version mismatch");
    }
    if (!supported_encoder(options->encoder)) {
        return fail(DAV2_STATUS_INVALID_ARGUMENT, "unsupported encoder");
    }
    return protect([&] {
        auto result = std::make_unique<dav2_context>();
        result->executor = dav2::create_executor(
            model_path_utf8, options->encoder, options->vulkan_device_index);
        *context = result.release();
    });
}

void DAV2_CALL dav2_destroy(dav2_context* context) {
    delete context;
}

dav2_status DAV2_CALL dav2_infer_tensor_f32(
    dav2_context* context,
    const float* normalized_rgb_chw,
    int32_t network_width,
    int32_t network_height,
    float* output_depth,
    size_t output_count) {
    if (context == nullptr || normalized_rgb_chw == nullptr || output_depth == nullptr) {
        return fail(DAV2_STATUS_INVALID_ARGUMENT, "null inference argument");
    }
    if (network_width <= 0 || network_height <= 0 ||
        network_width % 14 != 0 || network_height % 14 != 0) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "network dimensions must be positive multiples of 14");
    }
    const size_t required =
        static_cast<size_t>(network_width) * static_cast<size_t>(network_height);
    if (output_count < required) {
        return fail(DAV2_STATUS_BUFFER_TOO_SMALL, "output buffer is too small");
    }
    return protect([&] {
        context->executor->infer(
            normalized_rgb_chw, network_width, network_height, output_depth);
    });
}

dav2_status DAV2_CALL dav2_infer_bgr8(
    dav2_context* context,
    const uint8_t* bgr,
    int32_t image_width,
    int32_t image_height,
    ptrdiff_t row_stride_bytes,
    int32_t input_size,
    float* output_depth,
    size_t output_count) {
    if (context == nullptr || bgr == nullptr || output_depth == nullptr) {
        return fail(DAV2_STATUS_INVALID_ARGUMENT, "null inference argument");
    }
    const size_t required =
        image_width > 0 && image_height > 0
        ? static_cast<size_t>(image_width) * static_cast<size_t>(image_height)
        : 0;
    if (output_count < required) {
        return fail(DAV2_STATUS_BUFFER_TOO_SMALL, "output buffer is too small");
    }
    return protect([&] {
        const dav2::ImageShape shape =
            dav2::network_shape(image_width, image_height, input_size);
        dav2::preprocess_bgr8(
            bgr,
            image_width,
            image_height,
            row_stride_bytes,
            shape,
            context->image_scratch,
            context->image_input);
        context->executor->infer_resized(
            context->image_input.data(),
            shape.width,
            shape.height,
            output_depth,
            image_width,
            image_height);
    });
}

}  // extern "C"
