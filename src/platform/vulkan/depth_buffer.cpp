
#include "depth_buffer.h"
#include "command_pool.h"
#include "device.h"
#include "device_memory.h"
#include "image.h"
#include "image_view.h"
#include <core.h>
#include <vulkan/vulkan.hpp>

namespace vulkan {

VkFormat findSupportedFormat(const Device& device,
                             const std::vector<VkFormat>& candidates,
                             const VkImageTiling tiling,
                             const VkFormatFeatureFlags features) {
    for (auto format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(device.physicalDevice(), format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        }

        if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    LF_ASSERT(false, "failed to find supported format");
    return VkFormat {};
}

VkFormat findDepthFormat(const Device& device) {
    return findSupportedFormat(device,
                               {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                               VK_IMAGE_TILING_OPTIMAL,
                               VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

DepthBuffer::DepthBuffer(CommandPool& command_pool, const VkExtent2D extent) :
    format_(findDepthFormat(command_pool.device())) {
    const auto& device = command_pool.device();

    image_ = std::make_unique<Image>(device,
                                     extent,
                                     format_,
                                     VK_IMAGE_TILING_OPTIMAL,
                                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    image_memory_ = std::make_unique<DeviceMemory>(image_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    image_view_ = std::make_unique<class ImageView>(device, image_->handle(), format_, VK_IMAGE_ASPECT_DEPTH_BIT);

    image_->transitionImageLayout(command_pool, vk::ImageLayout::eDepthStencilAttachmentOptimal);
}

DepthBuffer::~DepthBuffer() {
    image_view_.reset();
    image_.reset();
    image_memory_.reset();  // release memory after bound image has been destroyed
}

}  // namespace vulkan
