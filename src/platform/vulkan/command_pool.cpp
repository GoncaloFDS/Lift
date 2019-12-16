#include "CommandPool.h"
#include "Device.h"

namespace Vulkan {

CommandPool::CommandPool(const class Device& device, const uint32_t queueFamilyIndex, const bool allowReset) :
	device_(device)
{
	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = queueFamilyIndex;
	poolInfo.flags = allowReset ? VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : 0;

    Vulkan_Check(vkCreateCommandPool(device.Handle(), &poolInfo, nullptr, &commandPool_),
                 "create command pool");
}

CommandPool::~CommandPool()
{
	if (commandPool_ != nullptr)
	{
		vkDestroyCommandPool(device_.Handle(), commandPool_, nullptr);
		commandPool_ = nullptr;
	}
}

}
