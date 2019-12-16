#pragma once

#include "VulkanError.h"

namespace vulkan {
class Device;

class DeviceMemory final {
public:

    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    DeviceMemory& operator=(DeviceMemory&&) = delete;

    DeviceMemory(const Device& device, size_t size, uint32_t memory_type_bits, VkMemoryPropertyFlags properties);
    DeviceMemory(DeviceMemory&& other) noexcept;
    ~DeviceMemory();

    [[nodiscard]] const class Device& device() const { return device_; }

    void* map(size_t offset, size_t size);
    void unmap();

private:

    [[nodiscard]] uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties) const;

    const class Device& device_;

VULKAN_HANDLE(VkDeviceMemory, memory_)
};

}
