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

}  // namespace

void VulkanContext::check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with Vulkan error " +
            std::to_string(static_cast<int>(result)));
    }
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
    vkGetPhysicalDeviceMemoryProperties(
        physical_device_, &memory_properties_);

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
        0,
        nullptr,
        nullptr,
    };
    check(
        vkCreateDevice(physical_device_, &device_info, nullptr, &device_),
        "vkCreateDevice");
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
    return create_buffer(
        bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

VulkanBuffer VulkanContext::create_host_buffer(VkDeviceSize bytes) {
    return create_buffer(
        bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
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

void VulkanContext::end_commands(VkCommandBuffer command) {
    check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    const VkSubmitInfo submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0,
        nullptr,
        nullptr,
        1,
        &command,
        0,
        nullptr,
    };
    check(vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
    check(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
    vkFreeCommandBuffers(device_, command_pool_, 1, &command);
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
    std::uint32_t group_z) {
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
        end_commands(command);
        pipeline.cached_descriptor_sets_.push_back(descriptor_set);
    }
}

void VulkanContext::destroy(VulkanBuffer& buffer) noexcept {
    if (batch_command_ != VK_NULL_HANDLE &&
        (buffer.buffer_ != VK_NULL_HANDLE ||
         buffer.memory_ != VK_NULL_HANDLE)) {
        batch_deferred_buffers_.push_back(
            {buffer.buffer_, buffer.memory_, buffer.mapped_});
        buffer.owner_ = nullptr;
        buffer.buffer_ = VK_NULL_HANDLE;
        buffer.memory_ = VK_NULL_HANDLE;
        buffer.mapped_ = nullptr;
        buffer.size_ = 0;
        return;
    }
    if (buffer.mapped_) {
        vkUnmapMemory(device_, buffer.memory_);
    }
    if (buffer.buffer_) {
        vkDestroyBuffer(device_, buffer.buffer_, nullptr);
    }
    if (buffer.memory_) {
        vkFreeMemory(device_, buffer.memory_, nullptr);
    }
    buffer.owner_ = nullptr;
    buffer.buffer_ = VK_NULL_HANDLE;
    buffer.memory_ = VK_NULL_HANDLE;
    buffer.mapped_ = nullptr;
    buffer.size_ = 0;
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
