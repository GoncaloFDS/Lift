#include "uniform_buffer.h"
#include "vulkan/buffer.h"

namespace assets {

UniformBuffer::UniformBuffer(const vulkan::Device& device) {
    const auto buffer_size = sizeof(UniformBufferObject);

    buffer_ = std::make_unique<vulkan::Buffer>(device, buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    memory_ = std::make_unique<vulkan::DeviceMemory>(
        buffer_->allocateMemory(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
}

UniformBuffer::UniformBuffer(UniformBuffer&& other) noexcept :
    buffer_(other.buffer_.release()), memory_(other.memory_.release()) {
}

UniformBuffer::~UniformBuffer() {
    buffer_.reset();
    memory_.reset();
}

void UniformBuffer::setValue(const UniformBufferObject& ubo) {
    const auto data = memory_->map(0, sizeof(UniformBufferObject));
    std::memcpy(data, &ubo, sizeof(ubo));
    memory_->unmap();
}

}  // namespace assets
