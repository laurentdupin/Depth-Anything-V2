#include "depth_anything_v2.h"

#include "executor.h"
#include "image.h"

#include <atomic>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct dav2_context {
    std::shared_ptr<dav2::Executor> executor;
    dav2::ImageScratch image_scratch;
    std::vector<float> image_input;
};

struct dav2_gpu_job {
    std::atomic<uint32_t> references{1};
    std::shared_ptr<dav2::Executor> executor;
    std::unique_ptr<dav2::GpuJob> implementation;
    uint64_t source_frame_id = 0;
};

struct dav2_gpu_output_lease {
    dav2_gpu_job* job = nullptr;
};

namespace {

thread_local std::string last_error;

class ApiError final : public std::runtime_error {
public:
    ApiError(dav2_status status, const char* message)
        : std::runtime_error(message), status_(status) {}
    dav2_status status() const { return status_; }

private:
    dav2_status status_;
};

dav2_status fail(dav2_status status, const char* message) {
    last_error = message ? message : "";
    return status;
}

bool supported_encoder(dav2_encoder encoder) {
    return encoder == DAV2_ENCODER_VITS ||
        encoder == DAV2_ENCODER_VITB ||
        encoder == DAV2_ENCODER_VITL;
}

void retain_gpu_job(dav2_gpu_job* job) {
    job->references.fetch_add(1, std::memory_order_relaxed);
}

void release_gpu_job(dav2_gpu_job* job) {
    if (job != nullptr &&
        job->references.fetch_sub(
            1, std::memory_order_acq_rel) == 1) {
        delete job;
    }
}

template <typename Function>
dav2_status protect(Function&& function) {
    try {
        function();
        last_error.clear();
        return DAV2_STATUS_OK;
    } catch (const ApiError& error) {
        return fail(error.status(), error.what());
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
GpuCapabilities probe_gpu_capabilities(int) {
    return {};
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
        case DAV2_STATUS_INVALID_STATE: return "invalid state";
        case DAV2_STATUS_CANCELLED: return "cancelled";
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
        result->executor = std::shared_ptr<dav2::Executor>(
            dav2::create_executor(
                model_path_utf8,
                options->encoder,
                options->vulkan_device_index));
        *context = result.release();
    });
}

void DAV2_CALL dav2_destroy(dav2_context* context) {
    delete context;
}

dav2_status DAV2_CALL dav2_probe_gpu_capabilities(
    int32_t vulkan_device_index,
    dav2_gpu_capabilities* capabilities) {
    if (capabilities == nullptr) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "GPU capabilities are null");
    }
    if (capabilities->struct_size < sizeof(*capabilities) ||
        vulkan_device_index < 0) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "invalid GPU capability probe");
    }
    return protect([&] {
        const dav2::GpuCapabilities available =
            dav2::probe_gpu_capabilities(vulkan_device_index);
        *capabilities = {};
        capabilities->struct_size = sizeof(*capabilities);
        capabilities->abi_version = DAV2_ABI_VERSION;
        capabilities->flags = available.flags;
        capabilities->adapter_luid = available.adapter_luid;
        capabilities->maximum_in_flight_jobs =
            available.maximum_in_flight_jobs;
    });
}

dav2_status DAV2_CALL dav2_get_gpu_capabilities(
    const dav2_context* context,
    dav2_gpu_capabilities* capabilities) {
    if (context == nullptr || capabilities == nullptr) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "null GPU capability argument");
    }
    if (capabilities->struct_size < sizeof(*capabilities)) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "GPU capabilities struct is too small");
    }
    return protect([&] {
        const dav2::GpuCapabilities available =
            context->executor->gpu_capabilities();
        *capabilities = {};
        capabilities->struct_size = sizeof(*capabilities);
        capabilities->abi_version = DAV2_ABI_VERSION;
        capabilities->flags = available.flags;
        capabilities->adapter_luid = available.adapter_luid;
        capabilities->maximum_in_flight_jobs =
            available.maximum_in_flight_jobs;
    });
}

dav2_status DAV2_CALL dav2_get_transfer_counters(
    const dav2_context* context,
    dav2_transfer_counters* counters) {
    if (context == nullptr || counters == nullptr) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "null transfer counter argument");
    }
    if (counters->struct_size < sizeof(*counters)) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "transfer counters struct is too small");
    }
    return protect([&] {
        std::uint64_t upload = 0;
        std::uint64_t download = 0;
        context->executor->transfer_counters(upload, download);
        *counters = {};
        counters->struct_size = sizeof(*counters);
        counters->tensor_upload_bytes = upload;
        counters->tensor_download_bytes = download;
    });
}

dav2_status DAV2_CALL dav2_submit_d3d12(
    dav2_context* context,
    const dav2_d3d12_submit_request* request,
    dav2_gpu_job** job) {
    if (context == nullptr || request == nullptr || job == nullptr) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "null D3D12 submit argument");
    }
    *job = nullptr;
    if (request->struct_size < sizeof(*request) ||
        request->abi_version != DAV2_ABI_VERSION) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "invalid D3D12 submit request");
    }
    return protect([&] {
        if (context->executor->gpu_capabilities().flags == 0) {
            throw ApiError(
                DAV2_STATUS_UNSUPPORTED,
                "complete D3D12/Vulkan GPU interop is unavailable");
        }
        dav2::GpuSubmitRequest native;
        native.shared_resource_handle =
            static_cast<std::uintptr_t>(
                request->shared_resource_handle);
        native.resource_byte_size = request->resource_byte_size;
        native.width = request->width;
        native.height = request->height;
        native.row_stride_bytes = request->row_stride_bytes;
        native.pixel_format =
            static_cast<dav2_gpu_pixel_format>(
                request->pixel_format);
        native.input_size = request->input_size;
        native.wait_fence_handle =
            static_cast<std::uintptr_t>(
                request->wait_fence_handle);
        native.wait_fence_value = request->wait_fence_value;
        native.source_frame_id = request->source_frame_id;
        native.timestamp_ns = request->timestamp_ns;
        auto result = std::make_unique<dav2_gpu_job>();
        result->executor = context->executor;
        result->source_frame_id = request->source_frame_id;
        result->implementation =
            context->executor->submit_gpu(native);
        *job = result.release();
    });
}

dav2_status DAV2_CALL dav2_submit_d3d12_texture(
    dav2_context* context,
    const dav2_d3d12_texture_submit_request* request,
    dav2_gpu_job** job) {
    if (context == nullptr || request == nullptr || job == nullptr) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "null D3D12 texture submit argument");
    }
    *job = nullptr;
    if (request->struct_size < sizeof(*request) ||
        request->abi_version != DAV2_ABI_VERSION) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "invalid D3D12 texture submit request");
    }
    return protect([&] {
        const std::uint64_t required =
            DAV2_GPU_CAP_D3D12_SHARED_TEXTURE_INPUT |
            DAV2_GPU_CAP_D3D12_SHARED_TEXTURE_OUTPUT;
        if ((context->executor->gpu_capabilities().flags &
             required) != required) {
            throw ApiError(
                DAV2_STATUS_UNSUPPORTED,
                "complete D3D12/Vulkan texture interop is unavailable");
        }
        dav2::GpuTextureSubmitRequest native;
        native.shared_texture_handle =
            static_cast<std::uintptr_t>(
                request->shared_texture_handle);
        native.width = request->width;
        native.height = request->height;
        native.pixel_format =
            static_cast<dav2_gpu_pixel_format>(
                request->pixel_format);
        native.input_size = request->input_size;
        native.wait_fence_handle =
            static_cast<std::uintptr_t>(
                request->wait_fence_handle);
        native.wait_fence_value = request->wait_fence_value;
        native.source_frame_id = request->source_frame_id;
        native.timestamp_ns = request->timestamp_ns;
        auto result = std::make_unique<dav2_gpu_job>();
        result->executor = context->executor;
        result->source_frame_id = request->source_frame_id;
        result->implementation =
            context->executor->submit_gpu_texture(native);
        *job = result.release();
    });
}

dav2_status DAV2_CALL dav2_gpu_job_poll(
    const dav2_gpu_job* job,
    dav2_gpu_job_status* status) {
    if (job == nullptr || status == nullptr) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "null GPU job poll argument");
    }
    if (status->struct_size < sizeof(*status)) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "GPU job status struct is too small");
    }
    return protect([&] {
        const dav2_gpu_job_state state =
            job->implementation->state();
        *status = {};
        status->struct_size = sizeof(*status);
        status->state = state;
        status->output_count =
            state == DAV2_GPU_JOB_CANCELLED ? 0u : 1u;
        status->source_frame_id = job->source_frame_id;
    });
}

dav2_status DAV2_CALL dav2_gpu_job_cancel(dav2_gpu_job* job) {
    if (job == nullptr) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "GPU job is null");
    }
    return protect([&] {
        if (job->implementation->state() ==
            DAV2_GPU_JOB_COMPLETE) {
            throw ApiError(
                DAV2_STATUS_INVALID_STATE,
                "completed GPU job cannot be cancelled");
        }
        job->implementation->cancel();
    });
}

void DAV2_CALL dav2_gpu_job_release(dav2_gpu_job* job) {
    release_gpu_job(job);
}

dav2_status DAV2_CALL dav2_gpu_output_acquire(
    dav2_gpu_job* job,
    uint32_t output_index,
    dav2_d3d12_output_descriptor* descriptor,
    dav2_gpu_output_lease** lease) {
    if (job == nullptr || descriptor == nullptr || lease == nullptr) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "null GPU output acquire argument");
    }
    *lease = nullptr;
    if (descriptor->struct_size < sizeof(*descriptor)) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "GPU output descriptor struct is too small");
    }
    if (output_index != 0) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "GPU output index does not exist");
    }
    return protect([&] {
        if (job->implementation->state() ==
            DAV2_GPU_JOB_CANCELLED) {
            throw ApiError(
                DAV2_STATUS_CANCELLED,
                "cancelled GPU job has no output");
        }
        const dav2::GpuOutput output =
            job->implementation->output();
        if (output.kind != dav2::GpuOutputKind::buffer) {
            throw ApiError(
                DAV2_STATUS_UNSUPPORTED,
                "GPU job output is a D3D12 texture");
        }
        auto result =
            std::make_unique<dav2_gpu_output_lease>();
        retain_gpu_job(job);
        result->job = job;
        *descriptor = {};
        descriptor->struct_size = sizeof(*descriptor);
        descriptor->abi_version = DAV2_ABI_VERSION;
        descriptor->pixel_format =
            DAV2_GPU_PIXEL_DEPTH_FLOAT32;
        descriptor->width = output.width;
        descriptor->height = output.height;
        descriptor->row_stride_bytes =
            output.row_stride_bytes;
        descriptor->byte_size = output.byte_size;
        descriptor->shared_resource_handle =
            output.shared_resource_handle;
        descriptor->ready_fence_handle =
            output.ready_fence_handle;
        descriptor->ready_fence_value =
            output.ready_fence_value;
        descriptor->source_frame_id =
            output.source_frame_id;
        descriptor->timestamp_ns = output.timestamp_ns;
        *lease = result.release();
    });
}

dav2_status DAV2_CALL dav2_gpu_texture_output_acquire(
    dav2_gpu_job* job,
    uint32_t output_index,
    dav2_d3d12_texture_output_descriptor* descriptor,
    dav2_gpu_output_lease** lease) {
    if (job == nullptr || descriptor == nullptr || lease == nullptr) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "null GPU texture output acquire argument");
    }
    *lease = nullptr;
    if (descriptor->struct_size < sizeof(*descriptor)) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "GPU texture output descriptor struct is too small");
    }
    if (output_index != 0) {
        return fail(
            DAV2_STATUS_INVALID_ARGUMENT,
            "GPU texture output index does not exist");
    }
    return protect([&] {
        if (job->implementation->state() ==
            DAV2_GPU_JOB_CANCELLED) {
            throw ApiError(
                DAV2_STATUS_CANCELLED,
                "cancelled GPU job has no texture output");
        }
        const dav2::GpuOutput output =
            job->implementation->output();
        if (output.kind != dav2::GpuOutputKind::texture) {
            throw ApiError(
                DAV2_STATUS_UNSUPPORTED,
                "GPU job output is a D3D12 buffer");
        }
        auto result =
            std::make_unique<dav2_gpu_output_lease>();
        retain_gpu_job(job);
        result->job = job;
        *descriptor = {};
        descriptor->struct_size = sizeof(*descriptor);
        descriptor->abi_version = DAV2_ABI_VERSION;
        descriptor->pixel_format =
            DAV2_GPU_PIXEL_DEPTH_FLOAT32;
        descriptor->width = output.width;
        descriptor->height = output.height;
        descriptor->shared_texture_handle =
            output.shared_resource_handle;
        descriptor->ready_fence_handle =
            output.ready_fence_handle;
        descriptor->ready_fence_value =
            output.ready_fence_value;
        descriptor->source_frame_id =
            output.source_frame_id;
        descriptor->timestamp_ns = output.timestamp_ns;
        *lease = result.release();
    });
}

void DAV2_CALL dav2_gpu_output_release(
    dav2_gpu_output_lease* lease) {
    if (lease == nullptr) return;
    release_gpu_job(lease->job);
    delete lease;
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
