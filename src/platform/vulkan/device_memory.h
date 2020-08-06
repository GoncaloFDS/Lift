#pragma once

#include "core/utilities.h"

namespace vulkan {
class Device;

class DeviceMemory final {
public:
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    DeviceMemory& operator=(DeviceMemory&&) = delete;

    DeviceMemory(const Device& device,
                 size_t size,
                 uint32_t memory_type_bits,
                 VkMemoryAllocateFlags allocate_flags,
                 VkMemoryPropertyFlags property_flags);
    DeviceMemory(DeviceMemory&& other) noexcept;
    ~DeviceMemory();

    [[nodiscard]] VkDeviceMemory handle() const { return memory_; }
    [[nodiscard]] const class Device& device() const { return device_; }

    void* map(size_t offset, size_t size);
    void unmap();

private:
    [[nodiscard]] uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags property_flags) const;

    const class Device& device_;
    VkDeviceMemory memory_ {};
};

}  // namespace vulkan
