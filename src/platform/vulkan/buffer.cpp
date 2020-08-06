#include "buffer.h"
#include "single_time_commands.h"

namespace vulkan {

Buffer::Buffer(const class Device& device, const size_t size, const VkBufferUsageFlags usage) : device_(device) {
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vulkanCheck(vkCreateBuffer(device.handle(), &buffer_info, nullptr, &buffer_), "create buffer");
}

Buffer::~Buffer() {
    if (buffer_ != nullptr) {
        vkDestroyBuffer(device_.handle(), buffer_, nullptr);
        buffer_ = nullptr;
    }
}

DeviceMemory Buffer::allocateMemory(const VkMemoryPropertyFlags property_flags) {
    return allocateMemory(0, property_flags);
}

DeviceMemory Buffer::allocateMemory(const VkMemoryAllocateFlags allocateFlags,
                                    const VkMemoryPropertyFlags propertyFlags) {
    const auto requirements = memoryRequirements();
    DeviceMemory memory(device_, requirements.size, requirements.memoryTypeBits, allocateFlags, propertyFlags);

    vulkanCheck(vkBindBufferMemory(device_.handle(), buffer_, memory.handle(), 0), "bind buffer memory");

    return memory;
}

VkMemoryRequirements Buffer::memoryRequirements() const {
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device_.handle(), buffer_, &requirements);
    return requirements;
}

void Buffer::copyFrom(CommandPool& command_pool, const Buffer& src, VkDeviceSize size) const {
    SingleTimeCommands::submit(command_pool, [&](VkCommandBuffer command_buffer) {
        VkBufferCopy copy_region = {};
        copy_region.srcOffset = 0;  // Optional
        copy_region.dstOffset = 0;  // Optional
        copy_region.size = size;

        vkCmdCopyBuffer(command_buffer, src.handle(), handle(), 1, &copy_region);
    });
}
VkDeviceAddress Buffer::deviceAddress() const {
    VkBufferDeviceAddressInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.pNext = nullptr;
    info.buffer = handle();

    return vkGetBufferDeviceAddress(device_.handle(), &info);
}

}  // namespace vulkan
