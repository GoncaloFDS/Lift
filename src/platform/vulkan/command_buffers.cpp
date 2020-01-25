#include "command_buffers.h"
#include "command_pool.h"
#include "device.h"

namespace vulkan {

CommandBuffers::CommandBuffers(CommandPool& command_pool, const uint32_t size) :
    command_pool_(command_pool) {
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool.handle();
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = size;

    command_buffers_.resize(size);

    vulkanCheck(vkAllocateCommandBuffers(command_pool.device().handle(), &alloc_info, command_buffers_.data()),
                "allocate command buffers");
}

CommandBuffers::~CommandBuffers() {
    if (!command_buffers_.empty()) {
        vkFreeCommandBuffers(command_pool_.device().handle(),
                             command_pool_.handle(),
                             static_cast<uint32_t>(command_buffers_.size()),
                             command_buffers_.data());
        command_buffers_.clear();
    }
}

VkCommandBuffer CommandBuffers::begin(const size_t i) {
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    begin_info.pInheritanceInfo = nullptr; // Optional

    vulkanCheck(vkBeginCommandBuffer(command_buffers_[i], &begin_info),
                "begin recording command buffer");

    return command_buffers_[i];
}

void CommandBuffers::end(const size_t i) {
    vulkanCheck(vkEndCommandBuffer(command_buffers_[i]),
                "record command buffer");
}

}
