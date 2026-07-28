#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dav2 {

class VulkanContext;

class VulkanBuffer {
public:
    VulkanBuffer() = default;
    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
    ~VulkanBuffer();

    VkBuffer handle() const { return buffer_; }
    VkDeviceSize size() const { return size_; }

private:
    friend class VulkanContext;
    VulkanContext* owner_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    void* mapped_ = nullptr;
};

class VulkanPipeline {
public:
    VulkanPipeline() = default;
    VulkanPipeline(VulkanPipeline&& other) noexcept;
    VulkanPipeline& operator=(VulkanPipeline&& other) noexcept;
    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;
    ~VulkanPipeline();

private:
    friend class VulkanContext;
    VulkanContext* owner_ = nullptr;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorType> descriptor_types_;
    std::uint32_t push_constant_bytes_ = 0;
};

class VulkanContext {
public:
    explicit VulkanContext(std::uint32_t device_index);
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    ~VulkanContext();

    const std::string& device_name() const { return device_name_; }

    VulkanBuffer create_device_buffer(VkDeviceSize bytes);
    VulkanBuffer create_host_buffer(VkDeviceSize bytes);
    void upload(VulkanBuffer& destination, const void* data, std::size_t bytes);
    void download(
        const VulkanBuffer& source,
        void* data,
        std::size_t bytes);

    VulkanPipeline create_pipeline(
        const std::uint32_t* spirv,
        std::size_t spirv_bytes,
        std::uint32_t binding_count,
        std::uint32_t push_constant_bytes);
    VulkanPipeline create_pipeline(
        const std::uint32_t* spirv,
        std::size_t spirv_bytes,
        const std::vector<VkDescriptorType>& descriptor_types,
        std::uint32_t push_constant_bytes);

    void dispatch(
        const VulkanPipeline& pipeline,
        const std::vector<const VulkanBuffer*>& buffers,
        const void* push_constants,
        std::uint32_t push_constant_bytes,
        std::uint32_t group_x,
        std::uint32_t group_y = 1,
        std::uint32_t group_z = 1);

private:
    friend class VulkanBuffer;
    friend class VulkanPipeline;

    std::uint32_t find_memory_type(
        std::uint32_t type_bits,
        VkMemoryPropertyFlags properties) const;
    VulkanBuffer create_buffer(
        VkDeviceSize bytes,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties);
    void copy_buffer(
        VkBuffer source,
        VkBuffer destination,
        VkDeviceSize bytes);
    VkCommandBuffer begin_commands();
    void end_commands(VkCommandBuffer command_buffer);
    void destroy(VulkanBuffer& buffer) noexcept;
    void destroy(VulkanPipeline& pipeline) noexcept;
    static void check(VkResult result, const char* operation);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    std::string device_name_;
};

}  // namespace dav2
