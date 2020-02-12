#pragma once

#include "core/utilities.h"

namespace vulkan {
class ImageMemoryBarrier final {
  public:
  static void insert(const VkCommandBuffer command_buffer, const VkImage image,
                     const VkImageSubresourceRange subresource_range, const VkAccessFlags src_access_mask,
                     const VkAccessFlags dst_access_mask, const VkImageLayout old_layout,
                     const VkImageLayout new_layout) {

    VkImageMemoryBarrier image_memory_barrier;
    image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_memory_barrier.pNext = nullptr;
    image_memory_barrier.srcAccessMask = src_access_mask;
    image_memory_barrier.dstAccessMask = dst_access_mask;
    image_memory_barrier.oldLayout = old_layout;
    image_memory_barrier.newLayout = new_layout;
    image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.image = image;
    image_memory_barrier.subresourceRange = subresource_range;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &image_memory_barrier);
  }
};

}  // namespace vulkan
