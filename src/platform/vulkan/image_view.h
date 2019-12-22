#pragma once

#include "core/utilities.h"

namespace vulkan {
class Device;

class ImageView final {
public:
    explicit ImageView(const Device& device, VkImage image, VkFormat format, VkImageAspectFlags aspect_flags);
    ~ImageView();

    [[nodiscard]] const Device& device() const { return device_; }

private:

    const class Device& device_;
    const VkImage image_;
    const VkFormat format_;

VULKAN_HANDLE(VkImageView, imageView_)
};

}
