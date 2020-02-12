#include "command_pool.h"
#include "device.h"

namespace vulkan {

CommandPool::CommandPool(const class Device &device, const uint32_t queue_family_index, const bool allow_reset) :
    device_(device) {
  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = queue_family_index;
  pool_info.flags = allow_reset ? VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : 0;

  vulkanCheck(vkCreateCommandPool(device.handle(), &pool_info, nullptr, &command_pool_), "create command pool");
}

CommandPool::~CommandPool() {
  if (command_pool_ != nullptr) {
    vkDestroyCommandPool(device_.handle(), command_pool_, nullptr);
    command_pool_ = nullptr;
  }
}

}  // namespace vulkan
