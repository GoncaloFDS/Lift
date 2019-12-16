#pragma once

#include "VulkanError.h"

namespace vulkan {
class ImageMemoryBarrier final {
public:

    static void insert(const VkCommandBuffer command_buffer,
                       const VkImage image,
                       const VkImageSubresourceRange subresource_range,
                       const VkAccessFlags src_access_mask,
                       const VkAccessFlags dst_access_mask,
                       const VkImageLayout old_layout,
                       const VkImageLayout new_layout) {

            VkImageMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.pNext = nullptr;
            barrier.srcAccessMask = src_access_mask;
            barrier.dstAccessMask = dst_access_mask;
            barrier.oldLayout = old_layout;
            barrier.newLayout = new_layout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = subresource_range;

            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                 &barrier);
    }
};

}
