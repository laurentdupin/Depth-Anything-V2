#include "vulkan.h"
#include "inferbridge/native_harness_vulkan_queue_priority.h"
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
#include <linux_capture_preprocess_spv.h>
#endif

#include "inferbridge/native_harness_cooperative_matrix.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dav2 {
namespace {

template <typename Handle>
void exchange_handle(Handle& left, Handle& right) {
    std::swap(left, right);
}

bool has_extension(
    const std::vector<VkExtensionProperties>& extensions,
    const char* name) {
    return std::any_of(
        extensions.begin(),
        extensions.end(),
        [name](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, name) == 0;
        });
}

}  // namespace

struct VulkanSubmission::Resources {
    std::vector<VulkanBatchedDescriptor> descriptor_sets;
    std::vector<VulkanDeferredBuffer> deferred_buffers;
    VulkanSemaphore wait;
    VulkanSemaphore signal;
    VkQueryPool profile_query_pool = VK_NULL_HANDLE;
    std::uint32_t profile_query_count = 0;
    std::vector<VulkanProfileRange> profile_ranges;
};

#include <inferbridge/vulkan/context/vulkan_context_check.h>

VulkanSemaphore::VulkanSemaphore(VulkanSemaphore&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_semaphore_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_semaphore_destroy_VulkanSemaphore.h>

VulkanSubmission::VulkanSubmission(VulkanSubmission&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_submission_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_submission_destroy_VulkanSubmission.h>

#include <inferbridge/vulkan/context/vulkan_submission_ready.h>

#include <inferbridge/vulkan/context/vulkan_submission_wait.h>

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_buffer_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_buffer_destroy_VulkanBuffer.h>

VulkanImage::VulkanImage(VulkanImage&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_image_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_image_destroy_VulkanImage.h>

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_pipeline_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_pipeline_destroy_VulkanPipeline.h>

#include <inferbridge/vulkan/context/vulkan_pipeline_set_debug_name.h>

VulkanContext::VulkanContext(
    std::uint32_t device_index,
    bool track_resource_hazards)
    : track_resource_hazards_(track_resource_hazards) {
    try {
    const VkApplicationInfo application{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "Depth Anything V2 C",
        1,
        "DAV2",
        1,
        VK_API_VERSION_1_3,
    };
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        &application,
        0,
        nullptr,
        0,
        nullptr,
    };
    check(vkCreateInstance(&instance_info, nullptr, &instance_), "vkCreateInstance");

    std::uint32_t device_count = 0;
    check(
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
        "vkEnumeratePhysicalDevices");
    if (device_count == 0 || device_index >= device_count) {
        throw std::runtime_error("requested Vulkan device does not exist");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    check(
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
        "vkEnumeratePhysicalDevices");
    physical_device_ = devices[device_index];

    std::uint32_t extension_count = 0;
    check(
        vkEnumerateDeviceExtensionProperties(
            physical_device_, nullptr, &extension_count, nullptr),
        "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(extension_count);
    check(
        vkEnumerateDeviceExtensionProperties(
            physical_device_, nullptr, &extension_count, extensions.data()),
        "vkEnumerateDeviceExtensionProperties");
    const bool has_cooperative_matrix = has_extension(
        extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    const bool has_cooperative_matrix2 = has_extension(
        extensions, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);

    VkPhysicalDeviceSubgroupProperties subgroup_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 base_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &subgroup_properties,
    };
    vkGetPhysicalDeviceProperties2(physical_device_, &base_properties);
    const VkPhysicalDeviceProperties& properties = base_properties.properties;
    device_name_ = properties.deviceName;
    compute_capabilities_.subgroup_arithmetic =
        (subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    VkPhysicalDeviceShaderIntegerDotProductFeatures integer_dot_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    };
    VkPhysicalDeviceShaderFloat16Int8Features float16_int8_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        &integer_dot_features,
    };
    VkPhysicalDevice8BitStorageFeatures storage8_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
        &float16_int8_features,
    };
    VkPhysicalDevice16BitStorageFeatures storage16_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        &storage8_features,
    };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        &storage16_features,
    };
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV cooperative2_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
        &cooperative_features,
    };
    VkPhysicalDeviceVulkanMemoryModelFeatures memory_model_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
        has_cooperative_matrix2
            ? static_cast<void*>(&cooperative2_features)
            : has_cooperative_matrix
            ? static_cast<void*>(&cooperative_features)
            : static_cast<void*>(&storage16_features),
    };
    VkPhysicalDeviceFeatures2 device_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        has_cooperative_matrix2
            ? static_cast<void*>(&memory_model_features)
            : has_cooperative_matrix
            ? static_cast<void*>(&memory_model_features)
            : static_cast<void*>(&storage16_features),
    };
    vkGetPhysicalDeviceFeatures2(physical_device_, &device_features);
    compute_capabilities_.fp16 =
        storage16_features.storageBuffer16BitAccess == VK_TRUE &&
        float16_int8_features.shaderFloat16 == VK_TRUE;
    VkPhysicalDeviceShaderIntegerDotProductProperties integer_dot_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 integer_dot_properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &integer_dot_properties,
    };
    vkGetPhysicalDeviceProperties2(
        physical_device_, &integer_dot_properties2);
    compute_capabilities_.packed_int8_dot =
        integer_dot_features.shaderIntegerDotProduct == VK_TRUE &&
        integer_dot_properties
            .integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;
    const auto subgroup_policy = inferbridge::native_harness::query_compute_subgroup_policy(physical_device_);
    if (has_cooperative_matrix &&
        cooperative_features.cooperativeMatrix == VK_TRUE &&
        memory_model_features.vulkanMemoryModel == VK_TRUE &&
        memory_model_features.vulkanMemoryModelDeviceScope == VK_TRUE &&
        (subgroup_properties.subgroupSize == 32 || subgroup_policy.required_size_32) &&
        (subgroup_properties.supportedStages &
         VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        compute_capabilities_.fp16) {
        const auto get_properties = reinterpret_cast<
            PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(
                instance_,
                "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
        if (get_properties != nullptr) {
            std::uint32_t count = 0;
            if (get_properties(physical_device_, &count, nullptr) ==
                VK_SUCCESS) {
                std::vector<VkCooperativeMatrixPropertiesKHR> matrices(
                    count,
                    {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
                if (get_properties(
                        physical_device_, &count, matrices.data()) ==
                    VK_SUCCESS) {
                    compute_capabilities_.cooperative_matrix_fp16 =
                        std::any_of(
                            matrices.begin(), matrices.end(),
                            [](const auto& matrix) {
                                return matrix.MSize == 16 &&
                                    matrix.NSize == 16 &&
                                    matrix.KSize == 16 &&
                                    matrix.AType ==
                                        VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    matrix.BType ==
                                        VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    matrix.CType ==
                                        VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    matrix.ResultType ==
                                        VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    matrix.scope ==
                                        VK_SCOPE_SUBGROUP_KHR;
                            });
                }
            }
        }
    }
    if (has_cooperative_matrix2 &&
        compute_capabilities_.cooperative_matrix_fp16 &&
        cooperative2_features.cooperativeMatrixWorkgroupScope == VK_TRUE &&
        cooperative2_features.cooperativeMatrixFlexibleDimensions == VK_TRUE &&
        cooperative2_features.cooperativeMatrixTensorAddressing == VK_TRUE &&
        memory_model_features.vulkanMemoryModel == VK_TRUE &&
        memory_model_features.vulkanMemoryModelDeviceScope == VK_TRUE) {
        const auto get_flexible_properties = reinterpret_cast<
            PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV>(
            vkGetInstanceProcAddr(
                instance_,
                "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV"));
        if (get_flexible_properties != nullptr) {
            std::uint32_t count = 0;
            if (get_flexible_properties(physical_device_, &count, nullptr) ==
                VK_SUCCESS) {
                std::vector<VkCooperativeMatrixFlexibleDimensionsPropertiesNV>
                    matrices(
                        count,
                        {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV});
                if (get_flexible_properties(
                        physical_device_, &count, matrices.data()) ==
                    VK_SUCCESS) {
                    compute_capabilities_.cooperative_matrix_workgroup_nv =
                        std::any_of(
                            matrices.begin(), matrices.end(),
                            [](const auto& matrix) {
                                return matrix.scope == VK_SCOPE_WORKGROUP_KHR &&
                                    matrix.workgroupInvocations == 128 &&
                                    matrix.MGranularity != 0 &&
                                    32 % matrix.MGranularity == 0 &&
                                    matrix.NGranularity != 0 &&
                                    32 % matrix.NGranularity == 0 &&
                                    matrix.KGranularity != 0 &&
                                    16 % matrix.KGranularity == 0 &&
                                    matrix.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    matrix.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    matrix.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    matrix.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR;
                            });
                    compute_capabilities_.cooperative_matrix_workgroup_int8_nv =
                        storage8_features.storageBuffer8BitAccess == VK_TRUE &&
                        float16_int8_features.shaderInt8 == VK_TRUE &&
                        std::any_of(
                            matrices.begin(), matrices.end(),
                            [](const auto& matrix) {
                                return matrix.scope == VK_SCOPE_WORKGROUP_KHR &&
                                    matrix.workgroupInvocations == 128 &&
                                    matrix.MGranularity != 0 &&
                                    32 % matrix.MGranularity == 0 &&
                                    matrix.NGranularity != 0 &&
                                    32 % matrix.NGranularity == 0 &&
                                    matrix.KGranularity != 0 &&
                                    32 % matrix.KGranularity == 0 &&
                                    matrix.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
                                    matrix.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
                                    matrix.CType == VK_COMPONENT_TYPE_SINT32_KHR &&
                                    matrix.ResultType == VK_COMPONENT_TYPE_SINT32_KHR;
                            });
                    const bool has_fp16_accumulator = std::any_of(
                        matrices.begin(), matrices.end(),
                        [](const auto& matrix) {
                            return matrix.scope == VK_SCOPE_WORKGROUP_KHR &&
                                matrix.workgroupInvocations == 128 &&
                                matrix.MGranularity != 0 &&
                                32 % matrix.MGranularity == 0 &&
                                matrix.NGranularity != 0 &&
                                32 % matrix.NGranularity == 0 &&
                                matrix.KGranularity != 0 &&
                                16 % matrix.KGranularity == 0 &&
                                matrix.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                matrix.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR;
                        });
                    compute_capabilities_
                        .cooperative_matrix_workgroup_epilogue_nv =
                            compute_capabilities_
                                .cooperative_matrix_workgroup_nv &&
                            has_fp16_accumulator &&
                            cooperative2_features
                                    .cooperativeMatrixConversions == VK_TRUE &&
                            cooperative2_features
                                    .cooperativeMatrixPerElementOperations ==
                                VK_TRUE;
                    compute_capabilities_.cooperative_matrix_workgroup_int8_nv =
                        compute_capabilities_.cooperative_matrix_workgroup_int8_nv &&
                        cooperative2_features.cooperativeMatrixConversions == VK_TRUE &&
                        cooperative2_features.cooperativeMatrixPerElementOperations ==
                            VK_TRUE;
                }
            }
        }
    }
    const auto cooperative_matrix_path =
        inferbridge::native::requested_cooperative_matrix_path();
    if (!inferbridge::native::cooperative_matrix_khr_allowed(
            cooperative_matrix_path)) {
        compute_capabilities_.cooperative_matrix_fp16 = false;
        compute_capabilities_.cooperative_matrix_workgroup_nv = false;
        compute_capabilities_.cooperative_matrix_workgroup_int8_nv = false;
        compute_capabilities_.cooperative_matrix_workgroup_epilogue_nv = false;
    } else if (!inferbridge::native::cooperative_matrix2_nv_allowed(
                   cooperative_matrix_path)) {
        compute_capabilities_.cooperative_matrix_workgroup_nv = false;
        compute_capabilities_.cooperative_matrix_workgroup_int8_nv = false;
        compute_capabilities_.cooperative_matrix_workgroup_epilogue_nv = false;
    }
    // Retain the DA V2-specific switches for existing diagnostics and scripts.
    if (const char* disabled =
            std::getenv("DAV2_DISABLE_COOPERATIVE_MATRIX")) {
        if (disabled[0] != '\0' && disabled[0] != '0') {
            compute_capabilities_.cooperative_matrix_fp16 = false;
            compute_capabilities_.cooperative_matrix_workgroup_nv = false;
            compute_capabilities_.cooperative_matrix_workgroup_int8_nv = false;
            compute_capabilities_.cooperative_matrix_workgroup_epilogue_nv = false;
        }
    }
    if (const char* disabled =
            std::getenv("DAV2_DISABLE_COOPERATIVE_MATRIX2")) {
        if (disabled[0] != '\0' && disabled[0] != '0') {
            compute_capabilities_.cooperative_matrix_workgroup_nv = false;
            compute_capabilities_.cooperative_matrix_workgroup_int8_nv = false;
            compute_capabilities_.cooperative_matrix_workgroup_epilogue_nv =
                false;
        }
    }
#if defined(_WIN32)
    VkPhysicalDeviceIDProperties identity{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &identity,
    };
    vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
    if (identity.deviceLUIDValid) {
        static_assert(VK_LUID_SIZE == sizeof(adapter_luid_));
        std::memcpy(
            &adapter_luid_, identity.deviceLUID, VK_LUID_SIZE);
    }
#endif
    vkGetPhysicalDeviceMemoryProperties(
        physical_device_, &memory_properties_);

    std::vector<const char*> enabled_extensions;
#if defined(__linux__) && !defined(__ANDROID__)
    linux_dma_buf_enabled_ = inferbridge::linux_capture::enable_extensions(
        extensions, enabled_extensions);
#endif
    if (compute_capabilities_.cooperative_matrix_fp16) {
        enabled_extensions.push_back(
            VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }
    if (compute_capabilities_.cooperative_matrix_workgroup_nv ||
        compute_capabilities_.cooperative_matrix_workgroup_int8_nv) {
        enabled_extensions.push_back(
            VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);
    }
#if defined(_WIN32)
    const bool has_external_memory_win32 = has_extension(
        extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    const bool has_external_semaphore_win32 = has_extension(
        extensions, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    if (has_external_memory_win32) {
        const VkPhysicalDeviceExternalBufferInfo external_buffer{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
            nullptr,
            0,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
        };
        VkExternalBufferProperties external_properties{
            VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
        };
        vkGetPhysicalDeviceExternalBufferProperties(
            physical_device_,
            &external_buffer,
            &external_properties);
        external_capabilities_.d3d12_resource_import =
            (external_properties.externalMemoryProperties
                 .externalMemoryFeatures &
             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
        const auto image_importable =
            [&](VkFormat format, VkImageUsageFlags usage) {
                const VkPhysicalDeviceExternalImageFormatInfo external{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
                    nullptr,
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
                };
                const VkPhysicalDeviceImageFormatInfo2 image_info{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                    &external,
                    format,
                    VK_IMAGE_TYPE_2D,
                    VK_IMAGE_TILING_OPTIMAL,
                    usage,
                    0,
                };
                VkExternalImageFormatProperties external_properties{
                    VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
                };
                VkImageFormatProperties2 image_properties{
                    VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                    &external_properties,
                };
                const VkResult queried =
                    vkGetPhysicalDeviceImageFormatProperties2(
                        physical_device_,
                        &image_info,
                        &image_properties);
                return queried == VK_SUCCESS &&
                    (external_properties.externalMemoryProperties
                         .externalMemoryFeatures &
                     VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
            };
        external_capabilities_
            .d3d12_bgra8_sampled_image_import =
            image_importable(
                VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        external_capabilities_
            .d3d12_rgba8_sampled_image_import =
            image_importable(
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        external_capabilities_
            .d3d12_r32_storage_image_import =
            image_importable(
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        enabled_extensions.push_back(
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    }
    if (has_external_semaphore_win32) {
        const VkPhysicalDeviceExternalSemaphoreInfo external_semaphore{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
            nullptr,
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
        };
        VkExternalSemaphoreProperties semaphore_properties{
            VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
        };
        vkGetPhysicalDeviceExternalSemaphoreProperties(
            physical_device_,
            &external_semaphore,
            &semaphore_properties);
        external_capabilities_.d3d12_fence_import =
            (semaphore_properties.externalSemaphoreFeatures &
             VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
        enabled_extensions.push_back(
            VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    }
#else
    (void)extensions;
#endif

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &family_count, families.data());
    auto family = inferbridge::native_harness::select_inference_queue_family(families, true);
    if (family == families.end()) {
        throw std::runtime_error("Vulkan device has no compute queue");
    }
    queue_family_ = static_cast<std::uint32_t>(
        std::distance(families.begin(), family));
    const char* profile_environment =
        std::getenv("DAV2_VULKAN_PROFILE");
    profile_dispatches_ =
        profile_environment != nullptr &&
        profile_environment[0] != '\0' &&
        profile_environment[0] != '0' &&
        family->timestampValidBits != 0;
    if (const char* batch_only =
            std::getenv("DAV2_VULKAN_PROFILE_BATCH_ONLY")) {
        profile_dispatch_ranges_ =
            batch_only[0] == '\0' || batch_only[0] == '0';
    }
    if (const char* skip =
            std::getenv("DAV2_VULKAN_PROFILE_SKIP_BATCHES")) {
        profile_skip_batches_ = std::strtoull(skip, nullptr, 10);
    }
    timestamp_period_ns_ = properties.limits.timestampPeriod;
    max_uniform_buffer_range_ = properties.limits.maxUniformBufferRange;

    // Inference is throughput work. Keep room for latency-sensitive rendering
    // when this independent Vulkan device shares a physical Android GPU with
    // Godot and the OpenXR compositor.
#if defined(__ANDROID__)
    constexpr float priority = 0.25f;
#else
    constexpr float priority = 1.0f;
#endif
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0,
        queue_family_,
        1,
        &priority,
    };
    VkPhysicalDeviceShaderIntegerDotProductFeatures enabled_integer_dot{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
        nullptr,
        compute_capabilities_.packed_int8_dot ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceShaderFloat16Int8Features enabled_float16_int8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        compute_capabilities_.packed_int8_dot
            ? &enabled_integer_dot : nullptr,
        compute_capabilities_.fp16 ? VK_TRUE : VK_FALSE,
        compute_capabilities_.cooperative_matrix_workgroup_int8_nv
            ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDevice8BitStorageFeatures enabled_storage8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
        &enabled_float16_int8,
        compute_capabilities_.cooperative_matrix_workgroup_int8_nv
            ? VK_TRUE : VK_FALSE,
        VK_FALSE,
        VK_FALSE,
    };
    VkPhysicalDevice16BitStorageFeatures enabled_storage16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        (compute_capabilities_.fp16 ||
         compute_capabilities_.packed_int8_dot ||
         compute_capabilities_.cooperative_matrix_workgroup_int8_nv)
            ? &enabled_storage8 : nullptr,
        compute_capabilities_.fp16 ? VK_TRUE : VK_FALSE,
        VK_FALSE,
        VK_FALSE,
        VK_FALSE,
    };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR enabled_cooperative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        (compute_capabilities_.fp16 ||
         compute_capabilities_.cooperative_matrix_workgroup_int8_nv)
            ? static_cast<void*>(&enabled_storage16)
            : compute_capabilities_.packed_int8_dot
            ? static_cast<void*>(&enabled_float16_int8)
            : nullptr,
        compute_capabilities_.cooperative_matrix_fp16 ? VK_TRUE : VK_FALSE,
        VK_FALSE,
    };
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV enabled_cooperative2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
        &enabled_cooperative,
        VK_TRUE,
        VK_TRUE,
        VK_FALSE,
        compute_capabilities_.cooperative_matrix_workgroup_epilogue_nv
            ? VK_TRUE : VK_FALSE,
        compute_capabilities_.cooperative_matrix_workgroup_epilogue_nv
            ? VK_TRUE : VK_FALSE,
        VK_TRUE,
        VK_FALSE,
    };
    VkPhysicalDeviceVulkanMemoryModelFeatures enabled_memory_model{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
        compute_capabilities_.cooperative_matrix_workgroup_nv
            ? static_cast<void*>(&enabled_cooperative2)
            : static_cast<void*>(&enabled_cooperative),
        VK_TRUE,
        VK_TRUE,
        VK_FALSE,
    };
    const void* enabled_feature_chain = compute_capabilities_.fp16
        ? (compute_capabilities_.cooperative_matrix_workgroup_nv
            ? static_cast<const void*>(&enabled_memory_model)
            : compute_capabilities_.cooperative_matrix_fp16
            ? static_cast<const void*>(&enabled_memory_model)
            : static_cast<const void*>(&enabled_storage16))
        : (compute_capabilities_.packed_int8_dot ||
           compute_capabilities_.cooperative_matrix_workgroup_int8_nv)
        ? static_cast<const void*>(&enabled_storage8)
        : nullptr;
    VkPhysicalDeviceSubgroupSizeControlFeatures enabled_subgroup_control{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
        const_cast<void*>(enabled_feature_chain),
        VK_TRUE,
        VK_FALSE,
    };
    const bool require_subgroup_control = compute_capabilities_.cooperative_matrix_fp16 &&
        subgroup_properties.subgroupSize != 32 && subgroup_policy.required_size_32;
    required_cooperative_subgroup_size_ = require_subgroup_control ? 32u : 0u;
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        require_subgroup_control ? &enabled_subgroup_control : enabled_feature_chain,
        0,
        1,
        &queue_info,
        0,
        nullptr,
        static_cast<std::uint32_t>(enabled_extensions.size()),
        enabled_extensions.data(),
        nullptr,
    };
    check(
        inferbridge::native_harness::create_inference_vulkan_device(physical_device_, &device_info, nullptr, &device_),
        "vkCreateDevice");
    std::fprintf(stderr, "InferBridge Vulkan compute: fp16=%u cooperative_matrix=%u subgroup=%u required_cooperative_subgroup=%u\n",
        compute_capabilities_.fp16 ? 1u : 0u, compute_capabilities_.cooperative_matrix_fp16 ? 1u : 0u,
        subgroup_properties.subgroupSize, require_subgroup_control ? 32u : 0u);
#if defined(_WIN32)
    get_memory_win32_handle_properties_ =
        reinterpret_cast<
            PFN_vkGetMemoryWin32HandlePropertiesKHR>(
            vkGetDeviceProcAddr(
                device_,
                "vkGetMemoryWin32HandlePropertiesKHR"));
    import_semaphore_win32_handle_ =
        reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(
                device_,
                "vkImportSemaphoreWin32HandleKHR"));
    external_capabilities_.d3d12_resource_import =
        external_capabilities_.d3d12_resource_import &&
        get_memory_win32_handle_properties_ != nullptr;
    external_capabilities_.d3d12_fence_import =
        external_capabilities_.d3d12_fence_import &&
        import_semaphore_win32_handle_ != nullptr;
#endif
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    const VkCommandPoolCreateInfo command_pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queue_family_,
    };
    check(
        vkCreateCommandPool(
            device_, &command_pool_info, nullptr, &command_pool_),
        "vkCreateCommandPool");

    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16384},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16384},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
    };
    const VkDescriptorPoolCreateInfo descriptor_pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        4096,
        static_cast<std::uint32_t>(std::size(pool_sizes)),
        pool_sizes,
    };
    check(
        vkCreateDescriptorPool(
            device_, &descriptor_pool_info, nullptr, &descriptor_pool_),
        "vkCreateDescriptorPool");
    if (profile_dispatches_) {
        const VkQueryPoolCreateInfo query_pool_info{
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            nullptr,
            0,
            VK_QUERY_TYPE_TIMESTAMP,
            2,
            0,
        };
        check(
            vkCreateQueryPool(
                device_,
                &query_pool_info,
                nullptr,
                &profile_query_pool_),
            "vkCreateQueryPool");
    }
    } catch (...) {
        release();
        throw;
    }
}

#include <inferbridge/vulkan/context/vulkan_context_destroy_VulkanContext.h>

#include <inferbridge/vulkan/context/vulkan_context_release_variant_1.h>

void VulkanContext::record_profile(
    const VulkanPipeline& pipeline,
    std::uint64_t ticks) {
    const std::string& name = pipeline.debug_name_.empty()
        ? std::string("unnamed")
        : pipeline.debug_name_;
    record_profile(name, ticks);
}

void VulkanContext::record_profile(
    const std::string& name,
    std::uint64_t ticks) {
    ProfileStat& stat = profile_stats_[name];
    stat.total_ticks += ticks;
    stat.maximum_ticks = std::max(stat.maximum_ticks, ticks);
    ++stat.dispatches;
}

void VulkanContext::collect_batch_profile(
    VkQueryPool query_pool,
    std::uint32_t query_count,
    const std::vector<VulkanProfileRange>& ranges) noexcept {
    if (query_pool == VK_NULL_HANDLE || query_count == 0) return;
    std::vector<std::uint64_t> timestamps(query_count);
    const VkResult result = vkGetQueryPoolResults(
        device_, query_pool, 0, query_count,
        timestamps.size() * sizeof(std::uint64_t), timestamps.data(),
        sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT |
            VK_QUERY_RESULT_WAIT_BIT);
    if (result == VK_SUCCESS) {
        for (const VulkanProfileRange& range : ranges) {
            if (range.begin < query_count && range.end < query_count &&
                timestamps[range.end] >= timestamps[range.begin]) {
                record_profile(
                    range.name,
                    timestamps[range.end] - timestamps[range.begin]);
            }
        }
    }
    vkDestroyQueryPool(device_, query_pool, nullptr);
}

void VulkanContext::print_profile() const noexcept {
    if (!profile_dispatches_ || profile_stats_.empty()) return;
    std::vector<std::pair<std::string, ProfileStat>> sorted(
        profile_stats_.begin(), profile_stats_.end());
    std::sort(
        sorted.begin(),
        sorted.end(),
        [](const auto& left, const auto& right) {
            return left.second.total_ticks >
                right.second.total_ticks;
        });
    std::fprintf(
        stderr,
        "DAV2 Vulkan profile: %s (GPU timestamps)\n",
        device_name_.c_str());
    for (const auto& [name, stat] : sorted) {
        const double total_ms =
            static_cast<double>(stat.total_ticks) *
            timestamp_period_ns_ / 1.0e6;
        const double average_ms =
            total_ms / static_cast<double>(stat.dispatches);
        const double maximum_ms =
            static_cast<double>(stat.maximum_ticks) *
            timestamp_period_ns_ / 1.0e6;
        std::fprintf(
            stderr,
            "  %-34s total=%10.3f ms count=%6llu "
            "avg=%8.4f ms max=%8.4f ms\n",
            name.c_str(),
            total_ms,
            static_cast<unsigned long long>(stat.dispatches),
            average_ms,
            maximum_ms);
        std::fprintf(
            stderr,
            "DAV2_VULKAN_PROFILE_JSON:{\"stage\":\"%s\","
            "\"total_ms\":%.6f,\"count\":%llu,"
            "\"average_ms\":%.6f,\"maximum_ms\":%.6f}\n",
            name.c_str(), total_ms,
            static_cast<unsigned long long>(stat.dispatches),
            average_ms, maximum_ms);
    }
}

#include <inferbridge/vulkan/context/vulkan_context_find_memory_type.h>

#include <inferbridge/vulkan/context/vulkan_context_create_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_create_device_buffer.h>

#if defined(_WIN32)
#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_image.h>

#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_fence.h>

#endif

#include <inferbridge/vulkan/context/vulkan_context_create_host_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_write_host.h>

#include <inferbridge/vulkan/context/vulkan_context_begin_commands.h>

#include <inferbridge/vulkan/context/vulkan_context_end_commands.h>

#include <inferbridge/vulkan/context/vulkan_context_submit_commands.h>

void VulkanContext::begin_batch() {
    if (batch_command_ != VK_NULL_HANDLE) {
        throw std::logic_error("nested Vulkan batch");
    }
    batch_command_ = begin_commands();
    batch_has_dispatch_ = false;
    batch_buffer_access_.clear();
    batch_image_access_.clear();
    batch_image_layout_.clear();
    batch_profile_query_count_ = 0;
    batch_profile_ranges_.clear();
    ++profile_batch_index_;
    if (profile_dispatches_ &&
        profile_batch_index_ > profile_skip_batches_) {
        constexpr std::uint32_t query_capacity = 8192;
        const VkQueryPoolCreateInfo query_pool_info{
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
            VK_QUERY_TYPE_TIMESTAMP, query_capacity, 0};
        check(vkCreateQueryPool(device_, &query_pool_info, nullptr,
                                &batch_profile_query_pool_),
              "vkCreateQueryPool(batch)");
        vkCmdResetQueryPool(
            batch_command_, batch_profile_query_pool_, 0, query_capacity);
        vkCmdWriteTimestamp(
            batch_command_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            batch_profile_query_pool_, batch_profile_query_count_++);
    }
}

#include <inferbridge/vulkan/context/vulkan_context_release_batch_resources.h>

VulkanSubmission VulkanContext::end_batch_async(
    VulkanSemaphore wait,
    VulkanSemaphore signal) {
    if (batch_command_ == VK_NULL_HANDLE) {
        throw std::logic_error("no active Vulkan batch");
    }
    auto resources = std::make_unique<VulkanSubmission::Resources>();
    resources->descriptor_sets = std::move(batch_descriptor_sets_);
    resources->deferred_buffers = std::move(batch_deferred_buffers_);
    resources->wait = std::move(wait);
    resources->signal = std::move(signal);
    if (batch_profile_query_pool_ != VK_NULL_HANDLE) {
        const std::uint32_t end = batch_profile_query_count_++;
        vkCmdWriteTimestamp(
            batch_command_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            batch_profile_query_pool_, end);
        resources->profile_query_pool = batch_profile_query_pool_;
        resources->profile_query_count = batch_profile_query_count_;
        resources->profile_ranges = std::move(batch_profile_ranges_);
        resources->profile_ranges.push_back({"batch_total", 0u, end});
        batch_profile_query_pool_ = VK_NULL_HANDLE;
        batch_profile_query_count_ = 0;
    }
    VkCommandBuffer command = batch_command_;
    batch_command_ = VK_NULL_HANDLE;
    batch_has_dispatch_ = false;
    try {
        VulkanSubmission result = submit_commands(
            command, &resources->wait, &resources->signal);
        result.resources_ = resources.release();
        return result;
    } catch (...) {
        for (const VulkanBatchedDescriptor& descriptor :
             resources->descriptor_sets) {
            if (descriptor.pipeline && descriptor.set) {
                descriptor.pipeline->cached_descriptor_sets_.push_back(
                    descriptor.set);
            }
        }
        for (const VulkanDeferredBuffer& buffer :
             resources->deferred_buffers) {
            recycle_or_destroy(buffer);
        }
        if (resources->profile_query_pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(
                device_, resources->profile_query_pool, nullptr);
            resources->profile_query_pool = VK_NULL_HANDLE;
        }
        throw;
    }
}

void VulkanContext::end_batch() {
    if (batch_command_ == VK_NULL_HANDLE) {
        throw std::logic_error("no active Vulkan batch");
    }
    VkQueryPool profile_pool = batch_profile_query_pool_;
    std::uint32_t profile_count = batch_profile_query_count_;
    std::vector<VulkanProfileRange> profile_ranges =
        std::move(batch_profile_ranges_);
    if (profile_pool != VK_NULL_HANDLE) {
        const std::uint32_t end = profile_count++;
        vkCmdWriteTimestamp(
            batch_command_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            profile_pool, end);
        profile_ranges.push_back({"batch_total", 0u, end});
        batch_profile_query_pool_ = VK_NULL_HANDLE;
        batch_profile_query_count_ = 0;
    }
    VkCommandBuffer command = batch_command_;
    batch_command_ = VK_NULL_HANDLE;
    batch_has_dispatch_ = false;
    end_commands(command);
    collect_batch_profile(profile_pool, profile_count, profile_ranges);
    release_batch_resources();
}

void VulkanContext::cancel_batch() noexcept {
    if (batch_command_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(
            device_, command_pool_, 1, &batch_command_);
        batch_command_ = VK_NULL_HANDLE;
    }
    if (batch_profile_query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(
            device_, batch_profile_query_pool_, nullptr);
        batch_profile_query_pool_ = VK_NULL_HANDLE;
    }
    batch_profile_query_count_ = 0;
    batch_profile_ranges_.clear();
    batch_has_dispatch_ = false;
    release_batch_resources();
}

#include <inferbridge/vulkan/context/vulkan_context_copy_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_upload_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_download_variant_3.h>

#include <inferbridge/vulkan/context/vulkan_context_transfer_counters.h>

#include <inferbridge/vulkan/context/vulkan_context_acquire_external_buffer_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_release_external_buffer_variant_2.h>

#include <inferbridge/linux_capture_vulkan_context.inl>


#include <inferbridge/vulkan/context/vulkan_context_acquire_external_image_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_release_external_image_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_4.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_3.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch_image_to_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch_buffer_to_image.h>

void VulkanContext::dispatch_resources(
    const VulkanPipeline& pipeline,
    const std::vector<VulkanDispatchResource>& resources,
    const void* push_constants,
    std::uint32_t push_constant_bytes,
    std::uint32_t group_x,
    std::uint32_t group_y,
    std::uint32_t group_z,
    const VulkanSemaphore* wait) {
    if (pipeline.owner_ != this)
        throw std::invalid_argument(
            "Vulkan dispatch uses a foreign pipeline: " +
            pipeline.debug_name_);
    if (resources.size() != pipeline.descriptor_types_.size())
        throw std::invalid_argument("Vulkan dispatch descriptor count mismatch");
    if (push_constant_bytes != pipeline.push_constant_bytes_)
        throw std::invalid_argument("Vulkan dispatch push constant size mismatch");
    if (push_constant_bytes && push_constants == nullptr)
        throw std::invalid_argument("Vulkan dispatch has null push constants");
    if (group_x == 0 || group_y == 0 || group_z == 0)
        throw std::invalid_argument("Vulkan dispatch has an empty workgroup dimension");

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    if (!pipeline.cached_descriptor_sets_.empty()) {
        descriptor_set = pipeline.cached_descriptor_sets_.back();
        pipeline.cached_descriptor_sets_.pop_back();
    } else {
        const VkDescriptorSetAllocateInfo allocate_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            descriptor_pool_,
            1,
            &pipeline.descriptor_layout_,
        };
        check(
            vkAllocateDescriptorSets(
                device_, &allocate_info, &descriptor_set),
            "vkAllocateDescriptorSets");
    }
    std::vector<VkDescriptorBufferInfo> buffer_info(resources.size());
    std::vector<VkDescriptorImageInfo> image_info(resources.size());
    std::vector<VkWriteDescriptorSet> writes(resources.size());
    for (std::size_t index = 0; index < resources.size(); ++index) {
        const VkDescriptorType type =
            pipeline.descriptor_types_[index];
        const bool image_descriptor =
            type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
            type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
            type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (image_descriptor) {
            const VulkanImage* image = resources[index].image;
            if (image == nullptr || image->owner_ != this ||
                image->view_ == VK_NULL_HANDLE ||
                (type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                 image->sampler_ == VK_NULL_HANDLE)) {
                pipeline.cached_descriptor_sets_.push_back(
                    descriptor_set);
                throw std::invalid_argument(
                    "foreign or invalid Vulkan image");
            }
            image_info[index] = {
                type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                    ? image->sampler_
                    : VK_NULL_HANDLE,
                image->view_,
                type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    ? VK_IMAGE_LAYOUT_GENERAL
                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            writes[index] = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                descriptor_set,
                static_cast<std::uint32_t>(index),
                0,
                1,
                type,
                &image_info[index],
                nullptr,
                nullptr,
            };
            continue;
        }
        const VulkanBuffer* buffer = resources[index].buffer;
        if (buffer == nullptr || buffer->owner_ != this) {
            pipeline.cached_descriptor_sets_.push_back(descriptor_set);
            throw std::invalid_argument("foreign Vulkan buffer");
        }
        buffer_info[index] = {
            buffer->buffer_,
            0,
            type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                    type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                ? std::min(buffer->size_, max_uniform_buffer_range_)
                : buffer->size_,
        };
        writes[index] = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptor_set,
            static_cast<std::uint32_t>(index),
            0,
            1,
            type,
            nullptr,
            &buffer_info[index],
            nullptr,
        };
    }
    vkUpdateDescriptorSets(
        device_,
        static_cast<std::uint32_t>(writes.size()),
        writes.data(),
        0,
        nullptr);

    const bool batched = batch_command_ != VK_NULL_HANDLE;
    if (batched && wait != nullptr) {
        pipeline.cached_descriptor_sets_.push_back(descriptor_set);
        throw std::invalid_argument(
            "external wait is not supported inside a Vulkan batch");
    }
    VkCommandBuffer command =
        batched ? batch_command_ : begin_commands();
    const bool profile =
        !batched && profile_query_pool_ != VK_NULL_HANDLE;
    std::uint32_t batch_profile_begin = 0;
    std::uint32_t batch_profile_end = 0;
    const bool profile_batch = batched &&
        profile_dispatch_ranges_ &&
        batch_profile_query_pool_ != VK_NULL_HANDLE &&
        batch_profile_query_count_ + 2u <= 8192u;
    if (profile_batch) {
        batch_profile_begin = batch_profile_query_count_++;
        batch_profile_end = batch_profile_query_count_++;
        vkCmdWriteTimestamp(
            command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            batch_profile_query_pool_, batch_profile_begin);
    }
    if (profile) {
        vkCmdResetQueryPool(
            command, profile_query_pool_, 0, 2);
        vkCmdWriteTimestamp(
            command,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            profile_query_pool_,
            0);
    }
    if (batched && batch_has_dispatch_ &&
        !track_resource_hazards_) {
        const VkMemoryBarrier barrier{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &barrier,
            0,
            nullptr,
            0,
            nullptr);
    } else if (batched) {
        std::unordered_map<VkBuffer, VkAccessFlags>
            current_buffers;
        std::unordered_map<VkImage, VkAccessFlags>
            current_images;
        std::unordered_map<VkImage, VkImageLayout> image_layouts;
        for (std::size_t index = 0;
             index < resources.size();
             ++index) {
            const VkAccessFlags access =
                pipeline.descriptor_access_[index];
            if (resources[index].buffer != nullptr) {
                current_buffers[
                    resources[index].buffer->buffer_] |= access;
            } else {
                const VkImage image =
                    resources[index].image->image_;
                current_images[image] |= access;
                image_layouts[image] =
                    pipeline.descriptor_types_[index] ==
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    ? VK_IMAGE_LAYOUT_GENERAL
                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        }
        std::vector<VkBufferMemoryBarrier> buffer_barriers;
        std::vector<VkImageMemoryBarrier> image_barriers;
        for (const auto& [buffer, access] : current_buffers) {
            const auto previous =
                batch_buffer_access_.find(buffer);
            if (previous != batch_buffer_access_.end() &&
                ((previous->second | access) &
                 VK_ACCESS_SHADER_WRITE_BIT) != 0) {
                buffer_barriers.push_back(
                    {
                        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        nullptr,
                        previous->second,
                        access,
                        VK_QUEUE_FAMILY_IGNORED,
                        VK_QUEUE_FAMILY_IGNORED,
                        buffer,
                        0,
                        VK_WHOLE_SIZE,
                    });
            }
            batch_buffer_access_[buffer] = access;
        }
        for (const auto& [image, access] : current_images) {
            const auto previous =
                batch_image_access_.find(image);
            const VkImageLayout layout = image_layouts[image];
            if (previous != batch_image_access_.end()) {
                const VkImageLayout previous_layout =
                    batch_image_layout_.at(image);
                if (((previous->second | access) &
                     VK_ACCESS_SHADER_WRITE_BIT) == 0 &&
                    previous_layout == layout) {
                    batch_image_access_[image] = access;
                    continue;
                }
                image_barriers.push_back(
                    {
                        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        nullptr,
                        previous->second,
                        access,
                        previous_layout,
                        layout,
                        VK_QUEUE_FAMILY_IGNORED,
                        VK_QUEUE_FAMILY_IGNORED,
                        image,
                        {
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            0,
                            1,
                            0,
                            1,
                        },
                    });
            }
            batch_image_access_[image] = access;
            batch_image_layout_[image] = layout;
        }
        if (!buffer_barriers.empty() ||
            !image_barriers.empty()) {
            vkCmdPipelineBarrier(
                command,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                nullptr,
                static_cast<std::uint32_t>(
                    buffer_barriers.size()),
                buffer_barriers.data(),
                static_cast<std::uint32_t>(
                    image_barriers.size()),
                image_barriers.data());
        }
    }
    vkCmdBindPipeline(
        command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_);
    vkCmdBindDescriptorSets(
        command,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.layout_,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);
    if (push_constant_bytes) {
        vkCmdPushConstants(
            command,
            pipeline.layout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            push_constant_bytes,
            push_constants);
    }
    vkCmdDispatch(command, group_x, group_y, group_z);
    if (profile_batch) {
        vkCmdWriteTimestamp(
            command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            batch_profile_query_pool_, batch_profile_end);
        batch_profile_ranges_.push_back({
            pipeline.debug_name_.empty() ? "unnamed" : pipeline.debug_name_,
            batch_profile_begin, batch_profile_end});
    }
    if (profile) {
        vkCmdWriteTimestamp(
            command,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            profile_query_pool_,
            1);
    }
    if (batched) {
        batch_has_dispatch_ = true;
        batch_descriptor_sets_.push_back(
            {const_cast<VulkanPipeline*>(&pipeline), descriptor_set});
    } else {
        end_commands(command, wait);
        if (profile) {
            std::uint64_t timestamps[2]{};
            check(
                vkGetQueryPoolResults(
                    device_,
                    profile_query_pool_,
                    0,
                    2,
                    sizeof(timestamps),
                    timestamps,
                    sizeof(std::uint64_t),
                    VK_QUERY_RESULT_64_BIT |
                        VK_QUERY_RESULT_WAIT_BIT),
                "vkGetQueryPoolResults");
            record_profile(
                pipeline, timestamps[1] - timestamps[0]);
        }
        pipeline.cached_descriptor_sets_.push_back(descriptor_set);
    }
}

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_4.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_5.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_2.h>

void VulkanContext::release_submission_resources(
    VulkanSubmission& submission) noexcept {
    if (submission.resources_ == nullptr) return;
    collect_batch_profile(
        submission.resources_->profile_query_pool,
        submission.resources_->profile_query_count,
        submission.resources_->profile_ranges);
    submission.resources_->profile_query_pool = VK_NULL_HANDLE;
    for (const VulkanBatchedDescriptor& descriptor :
         submission.resources_->descriptor_sets) {
        if (descriptor.pipeline && descriptor.set) {
            descriptor.pipeline->cached_descriptor_sets_.push_back(
                descriptor.set);
        }
    }
    for (const VulkanDeferredBuffer& buffer :
         submission.resources_->deferred_buffers) {
        recycle_or_destroy(buffer);
    }
    delete submission.resources_;
    submission.resources_ = nullptr;
}

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_recycle_or_destroy.h>

#include <inferbridge/vulkan/context/vulkan_context_discard.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_3.h>

}  // namespace dav2
