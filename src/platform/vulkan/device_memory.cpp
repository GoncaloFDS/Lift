#include "pch.h"
#include "device_memory.h"
#include "device.h"

namespace vulkan {

DeviceMemory::DeviceMemory(const class Device& device,
                           const size_t size,
                           const uint32_t memory_type_bits,
                           const VkMemoryPropertyFlags properties) : device_(device) {

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = size;
    alloc_info.memoryTypeIndex = findMemoryType(memory_type_bits, properties);

    vulkanCheck(vkAllocateMemory(device.Handle(), &alloc_info, nullptr, &memory_),
                "allocate memory");
}

DeviceMemory::DeviceMemory(DeviceMemory&& other) noexcept : device_(other.device_), memory_(other.memory_) {
    other.memory_ = nullptr;
}

DeviceMemory::~DeviceMemory() {
    if (memory_ != nullptr) {
        vkFreeMemory(device_.Handle(), memory_, nullptr);
        memory_ = nullptr;
    }
}

void* DeviceMemory::map(const size_t offset, const size_t size) {
    void* data;
    vulkanCheck(vkMapMemory(device_.Handle(), memory_, offset, size, 0, &data),
                "map memory");

    return data;
}

void DeviceMemory::unmap() {
    vkUnmapMemory(device_.Handle(), memory_);
}

uint32_t DeviceMemory::findMemoryType(const uint32_t type_filter, const VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(device_.physicalDevice(), &mem_properties);

    for (uint32_t i = 0; i != mem_properties.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    LF_ASSERT(false, "failed to find suitable memory type");
//	Throw(std::runtime_error("failed to find suitable memory type"));
}

}
