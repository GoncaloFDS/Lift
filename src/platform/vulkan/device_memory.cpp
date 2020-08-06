
#include "device_memory.h"
#include "device.h"
#include <core.h>

namespace vulkan {

DeviceMemory::DeviceMemory(const Device& device,
                           size_t size,
                           uint32_t memory_type_bits,
                           VkMemoryAllocateFlags allocate_flags,
                           VkMemoryPropertyFlags property_flags)
    : device_(device) {

    VkMemoryAllocateFlagsInfo flags_info = {};
    flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags_info.pNext = nullptr;
    flags_info.flags = allocate_flags;

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = nullptr;
    alloc_info.allocationSize = size;
    alloc_info.memoryTypeIndex = findMemoryType(memory_type_bits, property_flags);

    vulkanCheck(vkAllocateMemory(device.handle(), &alloc_info, nullptr, &memory_), "allocate memory");
}

DeviceMemory::DeviceMemory(DeviceMemory&& other) noexcept : device_(other.device_), memory_(other.memory_) {
    other.memory_ = nullptr;
}

DeviceMemory::~DeviceMemory() {
    if (memory_ != nullptr) {
        vkFreeMemory(device_.handle(), memory_, nullptr);
        memory_ = nullptr;
    }
}

void* DeviceMemory::map(const size_t offset, const size_t size) {
    void* data;
    vulkanCheck(vkMapMemory(device_.handle(), memory_, offset, size, 0, &data), "map memory");

    return data;
}

void DeviceMemory::unmap() {
    vkUnmapMemory(device_.handle(), memory_);
}

uint32_t DeviceMemory::findMemoryType(const uint32_t type_filter, const VkMemoryPropertyFlags property_flags) const {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(device_.physicalDevice(), &mem_properties);

    for (uint32_t i = 0; i != mem_properties.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & property_flags) == property_flags) {
            return i;
        }
    }

    LF_ASSERT(false, "failed to find suitable memory type");
    return 0;
}

}  // namespace vulkan
