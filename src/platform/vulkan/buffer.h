#pragma once

#include "VulkanError.h"
#include "device_memory.h"

namespace vulkan {
class CommandPool;
class Device;

class Buffer final {
public:
    Buffer(const Device& device, size_t size, VkBufferUsageFlags usage);
    ~Buffer();

    [[nodiscard]] const class Device& device() const { return device_; }

    DeviceMemory allocateMemory(VkMemoryPropertyFlags properties);
    [[nodiscard]] VkMemoryRequirements getMemoryRequirements() const;

    void copyFrom(CommandPool& command_pool, const Buffer& src, VkDeviceSize size);

private:

    const class Device& device_;

VULKAN_HANDLE(VkBuffer, buffer_)
};

}
