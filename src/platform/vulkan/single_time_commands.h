#pragma once

#include "command_buffers.h"
#include "command_pool.h"
#include "core/utilities.h"
#include "device.h"
#include <functional>

namespace vulkan {
class SingleTimeCommands final {
  public:
  static void submit(CommandPool &command_pool, const std::function<void(VkCommandBuffer)> &action) {
    CommandBuffers command_buffers(command_pool, 1);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(command_buffers[0], &begin_info);

    action(command_buffers[0]);

    vkEndCommandBuffer(command_buffers[0]);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffers[0];

    const auto graphics_queue = command_pool.device().graphicsQueue();

    vkQueueSubmit(graphics_queue, 1, &submit_info, nullptr);
    vkQueueWaitIdle(graphics_queue);
  }
};

}  // namespace vulkan
