#pragma once

#include "core/utilities.h"
#include "device_memory.h"

namespace vulkan {
class CommandPool;
class Device;

class Buffer final {
public:
    Buffer(const Device& device, size_t size, VkBufferUsageFlags usage);
    ~Buffer();

    [[nodiscard]] VkBuffer handle() const { return buffer_; }
    [[nodiscard]] const class Device& device() const { return device_; }

    DeviceMemory allocateMemory(const VkMemoryPropertyFlags property_flags);
    DeviceMemory allocateMemory(const VkMemoryAllocateFlags allocateFlags, const VkMemoryPropertyFlags propertyFlags);
    [[nodiscard]] VkMemoryRequirements memoryRequirements() const;
    [[nodiscard]] VkDeviceAddress deviceAddress() const;

    void copyFrom(CommandPool& command_pool, const Buffer& src, VkDeviceSize size) const;

private:
    const class Device& device_;
    VkBuffer buffer_ {};
};

}  // namespace vulkan
