#include "uniform_buffer.h"
#include "platform/vulkan/buffer.h"
#include <cstring>

namespace assets {

UniformBuffer::UniformBuffer(const vulkan::Device& device) {
    const auto bufferSize = sizeof(UniformBufferObject);

    buffer_.reset(new vulkan::Buffer(device, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    memory_.reset(new vulkan::DeviceMemory(buffer_->allocateMemory(
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)));
}

UniformBuffer::UniformBuffer(UniformBuffer&& other) noexcept :
    buffer_(other.buffer_.release()),
    memory_(other.memory_.release()) {
}

UniformBuffer::~UniformBuffer() {
    buffer_.reset();
    memory_.reset(); // release memory after bound buffer has been destroyed
}

void UniformBuffer::SetValue(const UniformBufferObject& ubo) {
    const auto data = memory_->map(0, sizeof(UniformBufferObject));
    std::memcpy(data, &ubo, sizeof(ubo));
    memory_->unmap();
}

}
