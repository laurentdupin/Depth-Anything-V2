#ifndef DEPTH_ANYTHING_V2_H
#define DEPTH_ANYTHING_V2_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(DAV2_BUILD_DLL)
#    define DAV2_API __declspec(dllexport)
#  else
#    define DAV2_API __declspec(dllimport)
#  endif
#  define DAV2_CALL __cdecl
#else
#  define DAV2_API __attribute__((visibility("default")))
#  define DAV2_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DAV2_ABI_VERSION 1u

typedef struct dav2_context dav2_context;
typedef struct dav2_gpu_job dav2_gpu_job;
typedef struct dav2_gpu_output_lease dav2_gpu_output_lease;

typedef enum dav2_status {
    DAV2_STATUS_OK = 0,
    DAV2_STATUS_INVALID_ARGUMENT = 1,
    DAV2_STATUS_MODEL_IO = 2,
    DAV2_STATUS_MODEL_FORMAT = 3,
    DAV2_STATUS_VULKAN_UNAVAILABLE = 4,
    DAV2_STATUS_OUT_OF_MEMORY = 5,
    DAV2_STATUS_INFERENCE_FAILED = 6,
    DAV2_STATUS_BUFFER_TOO_SMALL = 7,
    DAV2_STATUS_UNSUPPORTED = 8,
    DAV2_STATUS_INTERNAL_ERROR = 9,
    DAV2_STATUS_INVALID_STATE = 10,
    DAV2_STATUS_CANCELLED = 11
} dav2_status;

typedef enum dav2_encoder {
    DAV2_ENCODER_VITS = 0,
    DAV2_ENCODER_VITB = 1,
    DAV2_ENCODER_VITL = 2
} dav2_encoder;

typedef struct dav2_create_options {
    uint32_t struct_size;
    uint32_t abi_version;
    dav2_encoder encoder;
    int32_t vulkan_device_index;
    uint32_t flags;
} dav2_create_options;

enum {
    /*
     * Retain and execute FP32 weights and attention scores. This trades some
     * speed and device memory for the tightest cross-backend reproducibility.
     */
    DAV2_CREATE_FORCE_FP32 = 1u << 0u
};

typedef struct dav2_image_shape {
    int32_t width;
    int32_t height;
} dav2_image_shape;

enum {
    DAV2_GPU_CAP_D3D12_SHARED_INPUT = 1ull << 0u,
    DAV2_GPU_CAP_D3D12_FENCE_WAIT = 1ull << 1u,
    DAV2_GPU_CAP_D3D12_SHARED_OUTPUT = 1ull << 2u,
    DAV2_GPU_CAP_D3D12_FENCE_SIGNAL = 1ull << 3u,
    DAV2_GPU_CAP_ASYNC_SUBMIT = 1ull << 4u,
    DAV2_GPU_CAP_CANCELLATION = 1ull << 5u,
    DAV2_GPU_CAP_NO_HOST_PIXEL_STAGING = 1ull << 6u,
    DAV2_GPU_CAP_NO_HOST_DEPTH_STAGING = 1ull << 7u,
    DAV2_GPU_CAP_D3D12_SHARED_TEXTURE_INPUT = 1ull << 8u,
    DAV2_GPU_CAP_D3D12_SHARED_TEXTURE_OUTPUT = 1ull << 9u
};

typedef enum dav2_gpu_pixel_format {
    DAV2_GPU_PIXEL_BGRA8 = 1,
    DAV2_GPU_PIXEL_RGBA8 = 2,
    DAV2_GPU_PIXEL_DEPTH_FLOAT32 = 3
} dav2_gpu_pixel_format;

typedef enum dav2_gpu_job_state {
    DAV2_GPU_JOB_QUEUED = 0,
    DAV2_GPU_JOB_RUNNING = 1,
    DAV2_GPU_JOB_COMPLETE = 2,
    DAV2_GPU_JOB_FAILED = 3,
    DAV2_GPU_JOB_CANCELLED = 4
} dav2_gpu_job_state;

typedef struct dav2_gpu_capabilities {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t flags;
    uint64_t adapter_luid;
    uint32_t maximum_in_flight_jobs;
    uint32_t reserved;
} dav2_gpu_capabilities;

/*
 * Windows-only GPU submission contract.
 *
 * shared_resource_handle is a borrowed NT handle for a shared D3D12 buffer
 * containing tightly addressable BGRA8 or RGBA8 pixels. wait_fence_handle is
 * a borrowed NT handle for a shared D3D12 fence. DAV2 duplicates both handles
 * before returning from submit, so the caller may close them immediately.
 */
typedef struct dav2_d3d12_submit_request {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t shared_resource_handle;
    uint64_t resource_byte_size;
    uint32_t width;
    uint32_t height;
    uint32_t row_stride_bytes;
    uint32_t pixel_format;
    int32_t input_size;
    uint32_t reserved;
    uint64_t wait_fence_handle;
    uint64_t wait_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dav2_d3d12_submit_request;

/*
 * shared_texture_handle is a borrowed NT handle for a shared D3D12
 * TEXTURE2D. BGRA8 maps to DXGI_FORMAT_B8G8R8A8_UNORM and RGBA8 maps to
 * DXGI_FORMAT_R8G8B8A8_UNORM. DAV2 duplicates/imports the texture and fence
 * handles before returning.
 */
typedef struct dav2_d3d12_texture_submit_request {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t shared_texture_handle;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    int32_t input_size;
    uint64_t wait_fence_handle;
    uint64_t wait_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dav2_d3d12_texture_submit_request;

typedef struct dav2_gpu_job_status {
    uint32_t struct_size;
    uint32_t state;
    uint32_t output_count;
    uint32_t reserved;
    uint64_t source_frame_id;
} dav2_gpu_job_status;

/*
 * Handles in this descriptor are borrowed and remain valid until the matching
 * output lease is released. The resource is a D3D12 shared buffer containing
 * row-major float32 depth. Consumers must wait for ready_fence_value on the
 * shared D3D12 fence before accessing it. If a consumer transitions the
 * resource, it must return it to D3D12_RESOURCE_STATE_COMMON before releasing
 * the lease so DAV2 may safely reuse the slot.
 */
typedef struct dav2_d3d12_output_descriptor {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t pixel_format;
    uint32_t reserved;
    uint32_t width;
    uint32_t height;
    uint32_t row_stride_bytes;
    uint32_t reserved2;
    uint64_t byte_size;
    uint64_t shared_resource_handle;
    uint64_t ready_fence_handle;
    uint64_t ready_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dav2_d3d12_output_descriptor;

/*
 * The leased resource is a shared D3D12 TEXTURE2D with
 * DXGI_FORMAT_R32_FLOAT. Both handles are borrowed for the lease lifetime.
 * A consumer must return the texture to D3D12_RESOURCE_STATE_COMMON before
 * releasing the lease if it performs an explicit state transition.
 */
typedef struct dav2_d3d12_texture_output_descriptor {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t pixel_format;
    uint32_t reserved;
    uint32_t width;
    uint32_t height;
    uint32_t reserved2;
    uint32_t reserved3;
    uint64_t shared_texture_handle;
    uint64_t ready_fence_handle;
    uint64_t ready_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dav2_d3d12_texture_output_descriptor;

typedef struct dav2_transfer_counters {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t tensor_upload_bytes;
    uint64_t tensor_download_bytes;
} dav2_transfer_counters;

DAV2_API uint32_t DAV2_CALL dav2_abi_version(void);
DAV2_API const char* DAV2_CALL dav2_version_string(void);
DAV2_API const char* DAV2_CALL dav2_status_string(dav2_status status);
DAV2_API const char* DAV2_CALL dav2_last_error(void);

DAV2_API dav2_status DAV2_CALL dav2_get_network_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    dav2_image_shape* network_shape);

DAV2_API dav2_status DAV2_CALL dav2_get_inferbridge_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    dav2_image_shape* output_shape);

DAV2_API dav2_status DAV2_CALL dav2_create(
    const char* model_path_utf8,
    const dav2_create_options* options,
    dav2_context** context);

DAV2_API void DAV2_CALL dav2_destroy(dav2_context* context);

DAV2_API dav2_status DAV2_CALL dav2_probe_gpu_capabilities(
    int32_t vulkan_device_index,
    dav2_gpu_capabilities* capabilities);

DAV2_API dav2_status DAV2_CALL dav2_get_gpu_capabilities(
    const dav2_context* context,
    dav2_gpu_capabilities* capabilities);

DAV2_API dav2_status DAV2_CALL dav2_get_transfer_counters(
    const dav2_context* context,
    dav2_transfer_counters* counters);

DAV2_API dav2_status DAV2_CALL dav2_submit_d3d12(
    dav2_context* context,
    const dav2_d3d12_submit_request* request,
    dav2_gpu_job** job);

DAV2_API dav2_status DAV2_CALL dav2_submit_d3d12_texture(
    dav2_context* context,
    const dav2_d3d12_texture_submit_request* request,
    dav2_gpu_job** job);

DAV2_API dav2_status DAV2_CALL dav2_gpu_job_poll(
    const dav2_gpu_job* job,
    dav2_gpu_job_status* status);

DAV2_API dav2_status DAV2_CALL dav2_gpu_job_cancel(dav2_gpu_job* job);
DAV2_API void DAV2_CALL dav2_gpu_job_release(dav2_gpu_job* job);

DAV2_API dav2_status DAV2_CALL dav2_gpu_output_acquire(
    dav2_gpu_job* job,
    uint32_t output_index,
    dav2_d3d12_output_descriptor* descriptor,
    dav2_gpu_output_lease** lease);

DAV2_API dav2_status DAV2_CALL dav2_gpu_texture_output_acquire(
    dav2_gpu_job* job,
    uint32_t output_index,
    dav2_d3d12_texture_output_descriptor* descriptor,
    dav2_gpu_output_lease** lease);

DAV2_API void DAV2_CALL dav2_gpu_output_release(
    dav2_gpu_output_lease* lease);

/*
 * Runs the complete Python-compatible image path.
 *
 * Input is interleaved BGR uint8 data, matching cv2.imread. The output is one
 * row-major float32 depth value per source pixel. output_count must be at least
 * image_width * image_height.
 */
DAV2_API dav2_status DAV2_CALL dav2_infer_bgr8(
    dav2_context* context,
    const uint8_t* bgr,
    int32_t image_width,
    int32_t image_height,
    ptrdiff_t row_stride_bytes,
    int32_t input_size,
    float* output_depth,
    size_t output_count);

/*
 * Exact InferBridge BGRA contract: preserve captured BGR bytes as model
 * channels, use PyTorch legacy-nearest resize, retain the worker's transposed
 * interpolation dimensions, and return min/max-normalized FP32 depth at the
 * network dimensions.
 */
DAV2_API dav2_status DAV2_CALL dav2_inferbridge_bgra8_f32(
    dav2_context* context,
    const uint8_t* bgra,
    int32_t image_width,
    int32_t image_height,
    ptrdiff_t row_stride_bytes,
    int32_t input_size,
    float* output_depth,
    size_t output_count);

/*
 * Runs only the neural network. Input is normalized RGB planar float32 NCHW
 * data for a batch of one. Width and height must be positive multiples of 14.
 * Output has network_width * network_height float32 values.
 */
DAV2_API dav2_status DAV2_CALL dav2_infer_tensor_f32(
    dav2_context* context,
    const float* normalized_rgb_chw,
    int32_t network_width,
    int32_t network_height,
    float* output_depth,
    size_t output_count);

#ifdef __cplusplus
}
#endif

#endif
