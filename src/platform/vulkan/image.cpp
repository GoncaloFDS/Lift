#include "image.h"
#include "buffer.h"
#include "depth_buffer.h"
#include "device.h"
#include "single_time_commands.h"

namespace vulkan {

Image::Image(const class Device& device, const VkExtent2D extent, const VkFormat format) :
    Image(device,
          extent,
          format,
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT) {
}

Image::Image(const class Device& device,
             const VkExtent2D extent,
             const VkFormat format,
             const VkImageTiling tiling,
             const VkImageUsageFlags usage) :
    device_(device), extent_(extent), format_(format), image_layout_(VK_IMAGE_LAYOUT_UNDEFINED) {

    VkImageCreateInfo image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = extent.width;
    image_info.extent.height = extent.height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = format;
    image_info.tiling = tiling;
    image_info.initialLayout = image_layout_;
    image_info.usage = usage;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.flags = 0; // Optional

    vulkanCheck(vkCreateImage(device.handle(), &image_info, nullptr, &image_),
                "create image");
}

Image::Image(Image&& other) noexcept
    : device_(other.device_),
      extent_(other.extent_),
      format_(other.format_),
      image_layout_(other.image_layout_),
      image_(other.image_) {

    other.image_ = nullptr;
}

Image::~Image() {
    if (image_ != nullptr) {
        vkDestroyImage(device_.handle(), image_, nullptr);
        image_ = nullptr;
    }
}

DeviceMemory Image::allocateMemory(const VkMemoryPropertyFlags properties) const {
    const auto requirements = getMemoryRequirements();
    DeviceMemory memory(device_, requirements.size, requirements.memoryTypeBits, properties);

    vulkanCheck(vkBindImageMemory(device_.handle(), image_, memory.handle(), 0),
                "bind image memory");

    return memory;
}

VkMemoryRequirements Image::getMemoryRequirements() const {
    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(device_.handle(), image_, &requirements);
    return requirements;
}

void Image::transitionImageLayout(CommandPool& command_pool, VkImageLayout new_layout) {
    SingleTimeCommands::submit(command_pool, [&](VkCommandBuffer command_buffer) {
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = image_layout_;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image_;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

            if (DepthBuffer::hasStencilComponent(format_)) {
                barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        } else {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }

        VkPipelineStageFlags source_stage = 0;
        VkPipelineStageFlags destination_stage = 0;

        if (image_layout_ == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (image_layout_ == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
            && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else if (image_layout_ == VK_IMAGE_LAYOUT_UNDEFINED
            && new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask =
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        } else {
//			Throw(std::invalid_argument("unsupported layout transition"));
        }

        vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    });

    image_layout_ = new_layout;
}

void Image::copyFrom(CommandPool& command_pool, const Buffer& buffer) {
    SingleTimeCommands::submit(command_pool, [&](VkCommandBuffer command_buffer) {
        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {extent_.width, extent_.height, 1};

        vkCmdCopyBufferToImage(command_buffer,
                               buffer.handle(),
                               image_,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1,
                               &region);
    });
}

}
