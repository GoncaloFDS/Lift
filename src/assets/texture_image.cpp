#include "texture_image.h"
#include "texture.h"
#include "vulkan/buffer.h"
#include "vulkan/command_pool.h"
#include "vulkan/image.h"
#include "vulkan/image_view.h"
#include "vulkan/vulkan.hpp"

namespace assets {

TextureImage::TextureImage(vulkan::CommandPool& command_pool, const Texture& texture) {
    // Create a host staging buffer and copy the image into it.
    const VkDeviceSize image_size = texture.width() * texture.height() * 4;
    const auto& device = command_pool.device();

    auto staging_buffer = std::make_unique<vulkan::Buffer>(device, image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    auto staging_buffer_memory =
        staging_buffer->allocateMemory(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    const auto data = staging_buffer_memory.map(0, image_size);
    std::memcpy(data, texture.pixels(), image_size);
    staging_buffer_memory.unmap();

    // Create the device side image, memory, view and sampler.
    image_ = std::make_unique<vulkan::Image>(
        device,
        VkExtent2D {static_cast<uint32_t>(texture.width()), static_cast<uint32_t>(texture.height())},
        VK_FORMAT_R8G8B8A8_UNORM);
    image_memory_ = std::make_unique<vulkan::DeviceMemory>(image_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    image_view_ =
        std::make_unique<vulkan::ImageView>(device, image_->handle(), image_->format(), VK_IMAGE_ASPECT_COLOR_BIT);
    sampler_ = std::make_unique<vulkan::Sampler>(device, vulkan::SamplerConfig());

    // Transfer the data to device side.
    image_->transitionImageLayout(command_pool, vk::ImageLayout::eTransferDstOptimal);
    image_->copyFromBuffer(command_pool, vk::Buffer {staging_buffer->handle()});
    image_->transitionImageLayout(command_pool, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Delete the buffer before the memory
    staging_buffer.reset();
}

TextureImage::~TextureImage() {
    sampler_.reset();
    image_view_.reset();
    image_.reset();
    image_memory_.reset();
}

}  // namespace assets
