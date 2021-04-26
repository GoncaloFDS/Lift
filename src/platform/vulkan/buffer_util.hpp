
#include "buffer.h"
#include "command_pool.h"
#include "device.h"
#include "device_memory.h"
#include <cstring>
#include <memory>
#include <string>
#include <vector>
namespace vulkan {

class BufferUtil final {
public:
    template<class T>
    static void copyFromStagingBuffer(CommandPool& command_pool, Buffer& dst_buffer, const std::vector<T>& content);

    template<class T>
    static void createDeviceBuffer(CommandPool& command_pool,
                                   const char* const_name,
                                   const VkBufferUsageFlags usage,
                                   const std::vector<T>& content,
                                   std::unique_ptr<Buffer>& buffer,
                                   std::unique_ptr<DeviceMemory>& memory);
};

template<class T>
void BufferUtil::copyFromStagingBuffer(CommandPool& command_pool, Buffer& dst_buffer, const std::vector<T>& content) {
    const auto& device = command_pool.device();
    const auto content_size = sizeof(content[0]) * content.size();

    // Create a temporary host-visible staging buffer.
    auto staging_buffer = std::make_unique<Buffer>(device, content_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    auto staging_buffer_memory =
        staging_buffer->allocateMemory(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Copy the host data into the staging buffer.
    const auto data = staging_buffer_memory.map(0, content_size);
    std::memcpy(data, content.data(), content_size);
    staging_buffer_memory.unmap();

    // Copy the staging buffer to the device buffer.
    dst_buffer.copyFrom(command_pool, *staging_buffer, content_size);

    // Delete the buffer before the memory
    staging_buffer.reset();
}

template<class T>
void BufferUtil::createDeviceBuffer(CommandPool& command_pool,
                                    const char* const const_name,
                                    const VkBufferUsageFlags usage,
                                    const std::vector<T>& content,
                                    std::unique_ptr<Buffer>& buffer,
                                    std::unique_ptr<DeviceMemory>& memory) {
    const auto& device = command_pool.device();
    const auto content_size = sizeof(content[0]) * content.size();
    const VkMemoryAllocateFlags allocateFlags =
        usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;

    buffer = std::make_unique<Buffer>(device, content_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage);
    memory = std::make_unique<DeviceMemory>(buffer->allocateMemory(allocateFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    copyFromStagingBuffer(command_pool, *buffer, content);
}
}  // namespace vulkan