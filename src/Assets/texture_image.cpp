#include "texture_image.h"
#include "texture.h"
#include "platform/vulkan/buffer.h"
#include "platform/vulkan/command_pool.h"
#include "platform/vulkan/image_view.h"
#include "platform/vulkan/image.h"
#include "platform/vulkan/sampler.h"
#include <cstring>
#include <memory>

namespace assets {

TextureImage::TextureImage(vulkan::CommandPool& commandPool, const Texture& texture) {
    // Create a host staging buffer and copy the image into it.
    const VkDeviceSize imageSize = texture.Width() * texture.Height() * 4;
    const auto& device = commandPool.device();

    auto stagingBuffer = std::make_unique<vulkan::Buffer>(device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    auto stagingBufferMemory =
        stagingBuffer->allocateMemory(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    const auto data = stagingBufferMemory.map(0, imageSize);
    std::memcpy(data, texture.Pixels(), imageSize);
    stagingBufferMemory.unmap();

    // Create the device side image, memory, view and sampler.
    image_ = std::make_unique<vulkan::Image>(device,
                                             VkExtent2D{static_cast<uint32_t>(texture.Width()),
                                                        static_cast<uint32_t>(texture.Height())},
                                             VK_FORMAT_R8G8B8A8_UNORM);
    imageMemory_ = std::make_unique<vulkan::DeviceMemory>(image_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    imageView_ = std::make_unique<vulkan::ImageView>(device, image_->Handle(), image_->format(), VK_IMAGE_ASPECT_COLOR_BIT);
    sampler_ = std::make_unique<vulkan::Sampler>(device, vulkan::SamplerConfig());

    // Transfer the data to device side.
    image_->transitionImageLayout(commandPool, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    image_->copyFrom(commandPool, *stagingBuffer);
    image_->transitionImageLayout(commandPool, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Delete the buffer before the memory
    stagingBuffer.reset();
}

TextureImage::~TextureImage() {
    sampler_.reset();
    imageView_.reset();
    image_.reset();
    imageMemory_.reset();
}

}
