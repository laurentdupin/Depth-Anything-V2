#include "vulkan.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dav2 {
namespace {

template <typename Handle>
void exchange_handle(Handle& left, Handle& right) {
    std::swap(left, right);
}

#if defined(_WIN32)
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
#endif

}  // namespace

void VulkanContext::check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with Vulkan error " +
            std::to_string(static_cast<int>(result)));
    }
}

VulkanSemaphore::VulkanSemaphore(VulkanSemaphore&& other) noexcept {
    *this = std::move(other);
}

VulkanSemaphore& VulkanSemaphore::operator=(
    VulkanSemaphore&& other) noexcept {
    if (this != &other) {
        if (owner_) owner_->destroy(*this);
        owner_ = std::exchange(other.owner_, nullptr);
        semaphore_ = std::exchange(
            other.semaphore_, VK_NULL_HANDLE);
        value_ = std::exchange(other.value_, 0);
    }
    return *this;
}

VulkanSemaphore::~VulkanSemaphore() {
    if (owner_) owner_->destroy(*this);
}

VulkanSubmission::VulkanSubmission(VulkanSubmission&& other) noexcept {
    *this = std::move(other);
}

VulkanSubmission& VulkanSubmission::operator=(
    VulkanSubmission&& other) noexcept {
    if (this != &other) {
        if (owner_) owner_->destroy(*this);
        owner_ = std::exchange(other.owner_, nullptr);
        command_ = std::exchange(other.command_, VK_NULL_HANDLE);
        fence_ = std::exchange(other.fence_, VK_NULL_HANDLE);
    }
    return *this;
}

VulkanSubmission::~VulkanSubmission() {
    if (owner_) owner_->destroy(*this);
}

bool VulkanSubmission::ready() const {
    if (owner_ == nullptr || fence_ == VK_NULL_HANDLE) {
        throw std::logic_error("invalid Vulkan submission");
    }
    const VkResult result =
        vkGetFenceStatus(owner_->device_, fence_);
    if (result == VK_SUCCESS) return true;
    if (result == VK_NOT_READY) return false;
    VulkanContext::check(result, "vkGetFenceStatus");
    return false;
}

void VulkanSubmission::wait() const {
    if (owner_ == nullptr || fence_ == VK_NULL_HANDLE) {
        throw std::logic_error("invalid Vulkan submission");
    }
    VulkanContext::check(
        vkWaitForFences(
            owner_->device_, 1, &fence_, VK_TRUE, UINT64_MAX),
        "vkWaitForFences");
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept {
    *this = std::move(other);
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
    if (this != &other) {
        if (owner_) owner_->destroy(*this);
        owner_ = std::exchange(other.owner_, nullptr);
        buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
        memory_ = std::exchange(other.memory_, VK_NULL_HANDLE);
        size_ = std::exchange(other.size_, 0);
        mapped_ = std::exchange(other.mapped_, nullptr);
        cacheable_ = std::exchange(other.cacheable_, false);
    }
    return *this;
}

VulkanBuffer::~VulkanBuffer() {
    if (owner_) owner_->destroy(*this);
}

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept {
    *this = std::move(other);
}

VulkanPipeline& VulkanPipeline::operator=(VulkanPipeline&& other) noexcept {
    if (this != &other) {
        if (owner_) owner_->destroy(*this);
        owner_ = std::exchange(other.owner_, nullptr);
        descriptor_layout_ =
            std::exchange(other.descriptor_layout_, VK_NULL_HANDLE);
        layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
        pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
        descriptor_types_ = std::move(other.descriptor_types_);
        cached_descriptor_sets_ =
            std::move(other.cached_descriptor_sets_);
        push_constant_bytes_ = std::exchange(other.push_constant_bytes_, 0);
    }
    return *this;
}

VulkanPipeline::~VulkanPipeline() {
    if (owner_) owner_->destroy(*this);
}

VulkanContext::VulkanContext(std::uint32_t device_index) try {
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

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    device_name_ = properties.deviceName;
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
    std::vector<const char*> enabled_extensions;
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
    auto family = std::find_if(
        families.begin(), families.end(), [](const auto& candidate) {
            return candidate.queueCount > 0 &&
                (candidate.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        });
    if (family == families.end()) {
        throw std::runtime_error("Vulkan device has no compute queue");
    }
    queue_family_ = static_cast<std::uint32_t>(
        std::distance(families.begin(), family));

    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0,
        queue_family_,
        1,
        &priority,
    };
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        nullptr,
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
        vkCreateDevice(physical_device_, &device_info, nullptr, &device_),
        "vkCreateDevice");
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
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096},
    };
    const VkDescriptorPoolCreateInfo descriptor_pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        1024,
        2,
        pool_sizes,
    };
    check(
        vkCreateDescriptorPool(
            device_, &descriptor_pool_info, nullptr, &descriptor_pool_),
        "vkCreateDescriptorPool");
} catch (...) {
    release();
    throw;
}

VulkanContext::~VulkanContext() {
    release();
}

void VulkanContext::release() noexcept {
    if (device_) {
        vkDeviceWaitIdle(device_);
        cancel_batch();
        for (DeferredBuffer& buffer : device_buffer_pool_) {
            buffer.cacheable = false;
            recycle_or_destroy(buffer);
        }
        device_buffer_pool_.clear();
        pooled_device_bytes_ = 0;
        for (DeferredBuffer& buffer : host_buffer_pool_) {
            buffer.cacheable = false;
            recycle_or_destroy(buffer);
        }
        host_buffer_pool_.clear();
        pooled_host_bytes_ = 0;
        if (descriptor_pool_) {
            vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        }
        if (command_pool_) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
        }
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        descriptor_pool_ = VK_NULL_HANDLE;
        command_pool_ = VK_NULL_HANDLE;
    }
    if (instance_) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

std::uint32_t VulkanContext::find_memory_type(
    std::uint32_t type_bits,
    VkMemoryPropertyFlags properties) const {
    for (std::uint32_t index = 0;
         index < memory_properties_.memoryTypeCount;
         ++index) {
        if ((type_bits & (1u << index)) != 0 &&
            (memory_properties_.memoryTypes[index].propertyFlags & properties) ==
                properties) {
            return index;
        }
    }
    throw std::runtime_error("no compatible Vulkan memory type");
}

VulkanBuffer VulkanContext::create_buffer(
    VkDeviceSize bytes,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties) {
    if (bytes == 0) {
        throw std::invalid_argument("cannot create an empty Vulkan buffer");
    }
    VulkanBuffer result;
    result.owner_ = this;
    result.size_ = bytes;
    const VkBufferCreateInfo buffer_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        0,
        bytes,
        usage,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
    };
    check(
        vkCreateBuffer(device_, &buffer_info, nullptr, &result.buffer_),
        "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, result.buffer_, &requirements);
    const VkMemoryAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        requirements.size,
        find_memory_type(requirements.memoryTypeBits, properties),
    };
    check(
        vkAllocateMemory(device_, &allocate_info, nullptr, &result.memory_),
        "vkAllocateMemory");
    check(
        vkBindBufferMemory(device_, result.buffer_, result.memory_, 0),
        "vkBindBufferMemory");
    if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        check(
            vkMapMemory(
                device_, result.memory_, 0, bytes, 0, &result.mapped_),
            "vkMapMemory");
    }
    return result;
}

VulkanBuffer VulkanContext::create_device_buffer(VkDeviceSize bytes) {
    auto best = device_buffer_pool_.end();
    for (auto candidate = device_buffer_pool_.begin();
         candidate != device_buffer_pool_.end();
         ++candidate) {
        if (candidate->size >= bytes &&
            (best == device_buffer_pool_.end() ||
             candidate->size < best->size)) {
            best = candidate;
        }
    }
    if (best != device_buffer_pool_.end()) {
        VulkanBuffer result;
        result.owner_ = this;
        result.buffer_ = best->buffer;
        result.memory_ = best->memory;
        result.size_ = best->size;
        result.mapped_ = nullptr;
        result.cacheable_ = true;
        pooled_device_bytes_ -= best->size;
        device_buffer_pool_.erase(best);
        return result;
    }
    VulkanBuffer result = create_buffer(
        bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    result.cacheable_ = true;
    return result;
}

#if defined(_WIN32)
VulkanBuffer VulkanContext::import_d3d12_buffer(
    void* shared_handle,
    VkDeviceSize bytes) {
    if (!external_capabilities_.d3d12_resource_import) {
        throw std::runtime_error(
            "Vulkan device cannot import D3D12 resources");
    }
    if (shared_handle == nullptr || bytes == 0) {
        throw std::invalid_argument(
            "invalid D3D12 shared buffer");
    }
    HANDLE duplicated = nullptr;
    if (!DuplicateHandle(
            GetCurrentProcess(),
            static_cast<HANDLE>(shared_handle),
            GetCurrentProcess(),
            &duplicated,
            0,
            FALSE,
            DUPLICATE_SAME_ACCESS)) {
        throw std::runtime_error(
            "failed to duplicate D3D12 shared handle");
    }

    VulkanBuffer result;
    result.owner_ = this;
    result.size_ = bytes;
    const VkExternalMemoryBufferCreateInfo external_buffer{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        nullptr,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
    };
    const VkBufferCreateInfo buffer_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        &external_buffer,
        0,
        bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
    };
    try {
        check(
            vkCreateBuffer(
                device_, &buffer_info, nullptr, &result.buffer_),
            "vkCreateBuffer(D3D12 import)");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(
            device_, result.buffer_, &requirements);
        VkMemoryWin32HandlePropertiesKHR handle_properties{
            VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR,
        };
        check(
            get_memory_win32_handle_properties_(
                device_,
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
                duplicated,
                &handle_properties),
            "vkGetMemoryWin32HandlePropertiesKHR");
        const VkMemoryDedicatedAllocateInfo dedicated{
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
            nullptr,
            VK_NULL_HANDLE,
            result.buffer_,
        };
        const VkImportMemoryWin32HandleInfoKHR import{
            VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
            &dedicated,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
            duplicated,
            nullptr,
        };
        const VkMemoryAllocateInfo allocation{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            &import,
            requirements.size,
            find_memory_type(
                requirements.memoryTypeBits &
                    handle_properties.memoryTypeBits,
                0),
        };
        check(
            vkAllocateMemory(
                device_, &allocation, nullptr, &result.memory_),
            "vkAllocateMemory(D3D12 import)");
        CloseHandle(duplicated);
        duplicated = nullptr;
        check(
            vkBindBufferMemory(
                device_, result.buffer_, result.memory_, 0),
            "vkBindBufferMemory(D3D12 import)");
        return result;
    } catch (...) {
        if (duplicated != nullptr) CloseHandle(duplicated);
        throw;
    }
}

VulkanSemaphore VulkanContext::import_d3d12_fence(
    void* shared_handle,
    std::uint64_t value) {
    if (!external_capabilities_.d3d12_fence_import) {
        throw std::runtime_error(
            "Vulkan device cannot import D3D12 fences");
    }
    if (shared_handle == nullptr) {
        throw std::invalid_argument(
            "invalid D3D12 shared fence");
    }
    HANDLE duplicated = nullptr;
    if (!DuplicateHandle(
            GetCurrentProcess(),
            static_cast<HANDLE>(shared_handle),
            GetCurrentProcess(),
            &duplicated,
            0,
            FALSE,
            DUPLICATE_SAME_ACCESS)) {
        throw std::runtime_error(
            "failed to duplicate D3D12 fence handle");
    }
    VulkanSemaphore result;
    result.owner_ = this;
    result.value_ = value;
    const VkSemaphoreCreateInfo semaphore_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        nullptr,
        0,
    };
    try {
        check(
            vkCreateSemaphore(
                device_,
                &semaphore_info,
                nullptr,
                &result.semaphore_),
            "vkCreateSemaphore(D3D12 import)");
        const VkImportSemaphoreWin32HandleInfoKHR import{
            VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
            nullptr,
            result.semaphore_,
            0,
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
            duplicated,
            nullptr,
        };
        check(
            import_semaphore_win32_handle_(device_, &import),
            "vkImportSemaphoreWin32HandleKHR");
        CloseHandle(duplicated);
        return result;
    } catch (...) {
        CloseHandle(duplicated);
        throw;
    }
}
#endif

VulkanBuffer VulkanContext::create_host_buffer(VkDeviceSize bytes) {
    auto best = host_buffer_pool_.end();
    for (auto candidate = host_buffer_pool_.begin();
         candidate != host_buffer_pool_.end();
         ++candidate) {
        if (candidate->size >= bytes &&
            (best == host_buffer_pool_.end() ||
             candidate->size < best->size)) {
            best = candidate;
        }
    }
    if (best != host_buffer_pool_.end()) {
        VulkanBuffer result;
        result.owner_ = this;
        result.buffer_ = best->buffer;
        result.memory_ = best->memory;
        result.size_ = best->size;
        result.mapped_ = best->mapped;
        result.cacheable_ = true;
        pooled_host_bytes_ -= best->size;
        host_buffer_pool_.erase(best);
        return result;
    }
    VulkanBuffer result = create_buffer(
        bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    result.cacheable_ = true;
    return result;
}

void VulkanContext::write_host(
    VulkanBuffer& destination,
    const void* data,
    std::size_t bytes) {
    if (destination.owner_ != this || destination.mapped_ == nullptr ||
        data == nullptr || bytes > destination.size_) {
        throw std::invalid_argument("invalid Vulkan host write");
    }
    std::memcpy(destination.mapped_, data, bytes);
}

VkCommandBuffer VulkanContext::begin_commands() {
    const VkCommandBufferAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        command_pool_,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1,
    };
    VkCommandBuffer command = VK_NULL_HANDLE;
    check(
        vkAllocateCommandBuffers(device_, &allocate_info, &command),
        "vkAllocateCommandBuffers");
    const VkCommandBufferBeginInfo begin_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr,
    };
    check(vkBeginCommandBuffer(command, &begin_info), "vkBeginCommandBuffer");
    return command;
}

void VulkanContext::end_commands(
    VkCommandBuffer command,
    const VulkanSemaphore* wait) {
    VulkanSubmission submission = submit_commands(command, wait);
    submission.wait();
}

VulkanSubmission VulkanContext::submit_commands(
    VkCommandBuffer command,
    const VulkanSemaphore* wait) {
    if (wait != nullptr && wait->owner_ != this) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
        throw std::invalid_argument(
            "foreign Vulkan wait semaphore");
    }
    try {
        check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    } catch (...) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
        throw;
    }
    VkFence fence = VK_NULL_HANDLE;
    const VkFenceCreateInfo fence_info{
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        nullptr,
        0,
    };
    try {
        check(
            vkCreateFence(device_, &fence_info, nullptr, &fence),
            "vkCreateFence");
    } catch (...) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
        throw;
    }
    const VkPipelineStageFlags wait_stage =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
#if defined(_WIN32)
    const std::uint64_t wait_value =
        wait != nullptr ? wait->value_ : 0;
    const VkD3D12FenceSubmitInfoKHR d3d12_values{
        VK_STRUCTURE_TYPE_D3D12_FENCE_SUBMIT_INFO_KHR,
        nullptr,
        wait != nullptr ? 1u : 0u,
        wait != nullptr ? &wait_value : nullptr,
        0,
        nullptr,
    };
#endif
    const VkSemaphore wait_handle =
        wait != nullptr ? wait->semaphore_ : VK_NULL_HANDLE;
    const VkSubmitInfo submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
#if defined(_WIN32)
        wait != nullptr ? &d3d12_values : nullptr,
#else
        nullptr,
#endif
        wait != nullptr ? 1u : 0u,
        wait != nullptr ? &wait_handle : nullptr,
        wait != nullptr ? &wait_stage : nullptr,
        1,
        &command,
        0,
        nullptr,
    };
    const VkResult submitted =
        vkQueueSubmit(queue_, 1, &submit, fence);
    if (submitted != VK_SUCCESS) {
        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
        check(submitted, "vkQueueSubmit");
    }
    VulkanSubmission result;
    result.owner_ = this;
    result.command_ = command;
    result.fence_ = fence;
    return result;
}

void VulkanContext::begin_batch() {
    if (batch_command_ != VK_NULL_HANDLE) {
        throw std::logic_error("nested Vulkan batch");
    }
    batch_command_ = begin_commands();
    batch_has_dispatch_ = false;
}

void VulkanContext::release_batch_resources() noexcept {
    for (const BatchedDescriptor& descriptor :
         batch_descriptor_sets_) {
        if (descriptor.pipeline && descriptor.set) {
            descriptor.pipeline->cached_descriptor_sets_.push_back(
                descriptor.set);
        }
    }
    batch_descriptor_sets_.clear();
    for (const DeferredBuffer& buffer : batch_deferred_buffers_) {
        recycle_or_destroy(buffer);
    }
    batch_deferred_buffers_.clear();
}

void VulkanContext::end_batch() {
    if (batch_command_ == VK_NULL_HANDLE) {
        throw std::logic_error("no active Vulkan batch");
    }
    VkCommandBuffer command = batch_command_;
    batch_command_ = VK_NULL_HANDLE;
    batch_has_dispatch_ = false;
    end_commands(command);
    release_batch_resources();
}

void VulkanContext::cancel_batch() noexcept {
    if (batch_command_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(
            device_, command_pool_, 1, &batch_command_);
        batch_command_ = VK_NULL_HANDLE;
    }
    batch_has_dispatch_ = false;
    release_batch_resources();
}

void VulkanContext::copy_buffer(
    VkBuffer source,
    VkBuffer destination,
    VkDeviceSize bytes) {
    VkCommandBuffer command = begin_commands();
    const VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(command, source, destination, 1, &region);
    end_commands(command);
}

void VulkanContext::upload(
    VulkanBuffer& destination,
    const void* data,
    std::size_t bytes) {
    if (data == nullptr || bytes > destination.size_) {
        throw std::invalid_argument("invalid Vulkan upload");
    }
    VulkanBuffer staging = create_host_buffer(bytes);
    std::memcpy(staging.mapped_, data, bytes);
    copy_buffer(staging.buffer_, destination.buffer_, bytes);
}

void VulkanContext::download(
    const VulkanBuffer& source,
    void* data,
    std::size_t bytes) {
    if (data == nullptr || bytes > source.size_) {
        throw std::invalid_argument("invalid Vulkan download");
    }
    VulkanBuffer staging = create_host_buffer(bytes);
    copy_buffer(source.buffer_, staging.buffer_, bytes);
    std::memcpy(data, staging.mapped_, bytes);
}

VulkanPipeline VulkanContext::create_pipeline(
    const std::uint32_t* spirv,
    std::size_t spirv_bytes,
    std::uint32_t binding_count,
    std::uint32_t push_constant_bytes) {
    return create_pipeline(
        spirv,
        spirv_bytes,
        std::vector<VkDescriptorType>(
            binding_count, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
        push_constant_bytes);
}

VulkanPipeline VulkanContext::create_pipeline(
    const std::uint32_t* spirv,
    std::size_t spirv_bytes,
    const std::vector<VkDescriptorType>& descriptor_types,
    std::uint32_t push_constant_bytes) {
    if (spirv == nullptr || spirv_bytes == 0 || spirv_bytes % 4 != 0 ||
        descriptor_types.empty() ||
        push_constant_bytes > 128 ||
        push_constant_bytes % 4 != 0) {
        throw std::invalid_argument("invalid compute pipeline description");
    }
    VulkanPipeline result;
    result.owner_ = this;
    result.descriptor_types_ = descriptor_types;
    result.push_constant_bytes_ = push_constant_bytes;

    std::vector<VkDescriptorSetLayoutBinding> bindings(
        descriptor_types.size());
    for (std::uint32_t index = 0; index < descriptor_types.size(); ++index) {
        bindings[index] = {
            index,
            descriptor_types[index],
            1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr,
        };
    }
    const VkDescriptorSetLayoutCreateInfo descriptor_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        static_cast<std::uint32_t>(bindings.size()),
        bindings.data(),
    };
    check(
        vkCreateDescriptorSetLayout(
            device_, &descriptor_info, nullptr, &result.descriptor_layout_),
        "vkCreateDescriptorSetLayout");

    VkPushConstantRange push_range{
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        push_constant_bytes,
    };
    const VkPipelineLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        1,
        &result.descriptor_layout_,
        push_constant_bytes ? 1u : 0u,
        push_constant_bytes ? &push_range : nullptr,
    };
    check(
        vkCreatePipelineLayout(
            device_, &layout_info, nullptr, &result.layout_),
        "vkCreatePipelineLayout");

    const VkShaderModuleCreateInfo shader_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0,
        spirv_bytes,
        spirv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    check(
        vkCreateShaderModule(device_, &shader_info, nullptr, &shader),
        "vkCreateShaderModule");
    const VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        shader,
        "main",
        nullptr,
    };
    const VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr,
        0,
        stage,
        result.layout_,
        VK_NULL_HANDLE,
        -1,
    };
    try {
        check(
            vkCreateComputePipelines(
                device_,
                VK_NULL_HANDLE,
                1,
                &pipeline_info,
                nullptr,
                &result.pipeline_),
            "vkCreateComputePipelines");
    } catch (...) {
        vkDestroyShaderModule(device_, shader, nullptr);
        throw;
    }
    vkDestroyShaderModule(device_, shader, nullptr);
    return result;
}

void VulkanContext::dispatch(
    const VulkanPipeline& pipeline,
    const std::vector<const VulkanBuffer*>& buffers,
    const void* push_constants,
    std::uint32_t push_constant_bytes,
    std::uint32_t group_x,
    std::uint32_t group_y,
    std::uint32_t group_z,
    const VulkanSemaphore* wait) {
    if (pipeline.owner_ != this ||
        buffers.size() != pipeline.descriptor_types_.size() ||
        push_constant_bytes != pipeline.push_constant_bytes_ ||
        (push_constant_bytes && push_constants == nullptr) ||
        group_x == 0 || group_y == 0 || group_z == 0) {
        throw std::invalid_argument("invalid Vulkan dispatch");
    }

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
    std::vector<VkDescriptorBufferInfo> buffer_info(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        if (buffers[index] == nullptr || buffers[index]->owner_ != this) {
            pipeline.cached_descriptor_sets_.push_back(descriptor_set);
            throw std::invalid_argument("foreign Vulkan buffer");
        }
        buffer_info[index] = {
            buffers[index]->buffer_,
            0,
            buffers[index]->size_,
        };
        writes[index] = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptor_set,
            static_cast<std::uint32_t>(index),
            0,
            1,
            pipeline.descriptor_types_[index],
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
    if (batched && batch_has_dispatch_) {
        const VkMemoryBarrier barrier{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
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
    if (batched) {
        batch_has_dispatch_ = true;
        batch_descriptor_sets_.push_back(
            {const_cast<VulkanPipeline*>(&pipeline), descriptor_set});
    } else {
        end_commands(command, wait);
        pipeline.cached_descriptor_sets_.push_back(descriptor_set);
    }
}

void VulkanContext::destroy(VulkanBuffer& buffer) noexcept {
    if (batch_command_ != VK_NULL_HANDLE &&
        (buffer.buffer_ != VK_NULL_HANDLE ||
         buffer.memory_ != VK_NULL_HANDLE)) {
        batch_deferred_buffers_.push_back(
            {
                buffer.buffer_,
                buffer.memory_,
                buffer.mapped_,
                buffer.size_,
                buffer.cacheable_,
            });
        buffer.owner_ = nullptr;
        buffer.buffer_ = VK_NULL_HANDLE;
        buffer.memory_ = VK_NULL_HANDLE;
        buffer.mapped_ = nullptr;
        buffer.size_ = 0;
        buffer.cacheable_ = false;
        return;
    }
    recycle_or_destroy(
        {
            buffer.buffer_,
            buffer.memory_,
            buffer.mapped_,
            buffer.size_,
            buffer.cacheable_,
        });
    buffer.owner_ = nullptr;
    buffer.buffer_ = VK_NULL_HANDLE;
    buffer.memory_ = VK_NULL_HANDLE;
    buffer.mapped_ = nullptr;
    buffer.size_ = 0;
    buffer.cacheable_ = false;
}

void VulkanContext::destroy(VulkanSubmission& submission) noexcept {
    if (submission.fence_ != VK_NULL_HANDLE) {
        (void)vkWaitForFences(
            device_, 1, &submission.fence_, VK_TRUE, UINT64_MAX);
        vkDestroyFence(device_, submission.fence_, nullptr);
    }
    if (submission.command_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(
            device_, command_pool_, 1, &submission.command_);
    }
    submission.owner_ = nullptr;
    submission.command_ = VK_NULL_HANDLE;
    submission.fence_ = VK_NULL_HANDLE;
}

void VulkanContext::destroy(VulkanSemaphore& semaphore) noexcept {
    if (semaphore.semaphore_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(
            device_, semaphore.semaphore_, nullptr);
    }
    semaphore.owner_ = nullptr;
    semaphore.semaphore_ = VK_NULL_HANDLE;
    semaphore.value_ = 0;
}

void VulkanContext::recycle_or_destroy(
    DeferredBuffer buffer) noexcept {
    constexpr VkDeviceSize maximum_device_pool_bytes =
        VkDeviceSize{2} * 1024 * 1024 * 1024;
    constexpr VkDeviceSize maximum_host_pool_bytes =
        VkDeviceSize{64} * 1024 * 1024;
    if (buffer.cacheable && buffer.buffer && buffer.memory) {
        if (buffer.mapped &&
            buffer.size <= maximum_host_pool_bytes - pooled_host_bytes_) {
            pooled_host_bytes_ += buffer.size;
            host_buffer_pool_.push_back(buffer);
            return;
        }
        if (!buffer.mapped &&
            buffer.size <=
                maximum_device_pool_bytes - pooled_device_bytes_) {
            pooled_device_bytes_ += buffer.size;
            device_buffer_pool_.push_back(buffer);
            return;
        }
    }
    if (buffer.mapped) {
        vkUnmapMemory(device_, buffer.memory);
    }
    if (buffer.buffer) {
        vkDestroyBuffer(device_, buffer.buffer, nullptr);
    }
    if (buffer.memory) {
        vkFreeMemory(device_, buffer.memory, nullptr);
    }
}

void VulkanContext::discard(VulkanBuffer& buffer) noexcept {
    if (buffer.owner_ != this || buffer.buffer_ == VK_NULL_HANDLE) {
        return;
    }
    if (batch_command_ != VK_NULL_HANDLE) {
        destroy(buffer);
        return;
    }
    if (buffer.mapped_) {
        vkUnmapMemory(device_, buffer.memory_);
    }
    vkDestroyBuffer(device_, buffer.buffer_, nullptr);
    vkFreeMemory(device_, buffer.memory_, nullptr);
    buffer.owner_ = nullptr;
    buffer.buffer_ = VK_NULL_HANDLE;
    buffer.memory_ = VK_NULL_HANDLE;
    buffer.size_ = 0;
    buffer.mapped_ = nullptr;
    buffer.cacheable_ = false;
}

void VulkanContext::destroy(VulkanPipeline& pipeline) noexcept {
    if (!pipeline.cached_descriptor_sets_.empty()) {
        vkFreeDescriptorSets(
            device_,
            descriptor_pool_,
            static_cast<std::uint32_t>(
                pipeline.cached_descriptor_sets_.size()),
            pipeline.cached_descriptor_sets_.data());
        pipeline.cached_descriptor_sets_.clear();
    }
    if (pipeline.pipeline_) {
        vkDestroyPipeline(device_, pipeline.pipeline_, nullptr);
    }
    if (pipeline.layout_) {
        vkDestroyPipelineLayout(device_, pipeline.layout_, nullptr);
    }
    if (pipeline.descriptor_layout_) {
        vkDestroyDescriptorSetLayout(
            device_, pipeline.descriptor_layout_, nullptr);
    }
    pipeline.owner_ = nullptr;
    pipeline.pipeline_ = VK_NULL_HANDLE;
    pipeline.layout_ = VK_NULL_HANDLE;
    pipeline.descriptor_layout_ = VK_NULL_HANDLE;
    pipeline.descriptor_types_.clear();
}

}  // namespace dav2
