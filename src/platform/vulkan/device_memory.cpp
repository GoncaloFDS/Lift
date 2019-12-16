#include "DeviceMemory.h"
#include "device.h"

namespace Vulkan {

DeviceMemory::DeviceMemory(
	const class Device& device, 
	const size_t size, 
	const uint32_t memoryTypeBits, 
	const VkMemoryPropertyFlags properties) :
	device_(device)
{
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = size;
	allocInfo.memoryTypeIndex = FindMemoryType(memoryTypeBits, properties);

	Vulkan_Check(vkAllocateMemory(device.Handle(), &allocInfo, nullptr, &memory_),
				 "allocate memory");
}

DeviceMemory::DeviceMemory(DeviceMemory&& other) noexcept :
	device_(other.device_),
	memory_(other.memory_)
{
	other.memory_ = nullptr;
}

DeviceMemory::~DeviceMemory()
{
	if (memory_ != nullptr)
	{
		vkFreeMemory(device_.Handle(), memory_, nullptr);
		memory_ = nullptr;
	}
}

void* DeviceMemory::Map(const size_t offset, const size_t size)
{
	void* data;
	Vulkan_Check(vkMapMemory(device_.Handle(), memory_, offset, size, 0, &data),
				 "map memory");

	return data;
}

void DeviceMemory::Unmap()
{
	vkUnmapMemory(device_.Handle(), memory_);
}

uint32_t DeviceMemory::FindMemoryType(const uint32_t typeFilter, const VkMemoryPropertyFlags properties) const
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(device_.PhysicalDevice(), &memProperties);

	for (uint32_t i = 0; i != memProperties.memoryTypeCount; ++i)
	{
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}

//	Throw(std::runtime_error("failed to find suitable memory type"));
}

}
