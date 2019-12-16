#include "buffer.h"
#include "single_time_commands.h"

namespace vulkan {

Buffer::Buffer(const class Device& device, const size_t size, const VkBufferUsageFlags usage) :
    device_(device) {
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vulkanCheck(vkCreateBuffer(device.Handle(), &buffer_info, nullptr, &buffer_),
                "create buffer");
}

Buffer::~Buffer() {
    if (buffer_ != nullptr) {
        vkDestroyBuffer(device_.Handle(), buffer_, nullptr);
        buffer_ = nullptr;
    }
}

DeviceMemory Buffer::allocateMemory(const VkMemoryPropertyFlags properties) {
    const auto requirements = getMemoryRequirements();
    DeviceMemory memory(device_, requirements.size, requirements.memoryTypeBits, properties);

    vulkanCheck(vkBindBufferMemory(device_.Handle(), buffer_, memory.Handle(), 0),
                "bind buffer memory");

    return memory;
}

VkMemoryRequirements Buffer::getMemoryRequirements() const {
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device_.Handle(), buffer_, &requirements);
    return requirements;
}

void Buffer::copyFrom(CommandPool& command_pool, const Buffer& src, VkDeviceSize size) {
    SingleTimeCommands::submit(command_pool, [&](VkCommandBuffer command_buffer) {
        VkBufferCopy copy_region = {};
        copy_region.srcOffset = 0; // Optional
        copy_region.dstOffset = 0; // Optional
        copy_region.size = size;

        vkCmdCopyBuffer(command_buffer, src.Handle(), Handle(), 1, &copy_region);
    });
}

}
