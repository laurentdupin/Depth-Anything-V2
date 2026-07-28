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
    DAV2_STATUS_INTERNAL_ERROR = 9
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

typedef struct dav2_image_shape {
    int32_t width;
    int32_t height;
} dav2_image_shape;

DAV2_API uint32_t DAV2_CALL dav2_abi_version(void);
DAV2_API const char* DAV2_CALL dav2_version_string(void);
DAV2_API const char* DAV2_CALL dav2_status_string(dav2_status status);
DAV2_API const char* DAV2_CALL dav2_last_error(void);

DAV2_API dav2_status DAV2_CALL dav2_get_network_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    dav2_image_shape* network_shape);

DAV2_API dav2_status DAV2_CALL dav2_create(
    const char* model_path_utf8,
    const dav2_create_options* options,
    dav2_context** context);

DAV2_API void DAV2_CALL dav2_destroy(dav2_context* context);

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
