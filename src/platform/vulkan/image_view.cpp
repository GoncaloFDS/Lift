#include "image_view.h"
#include "device.h"

namespace vulkan {

ImageView::ImageView(const class Device& device,
                     const VkImage image,
                     const VkFormat format,
                     const VkImageAspectFlags aspect_flags) :
    device_(device),
    image_(image),
    format_(format) {
    VkImageViewCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    create_info.image = image;
    create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    create_info.format = format;
    create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.subresourceRange.aspectMask = aspect_flags;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;

    vulkanCheck(vkCreateImageView(device_.Handle(), &create_info, nullptr, &imageView_),
                "create image view");
}

ImageView::~ImageView() {
    if (imageView_ != nullptr) {
        vkDestroyImageView(device_.Handle(), imageView_, nullptr);
        imageView_ = nullptr;
    }
}

}