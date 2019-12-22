#pragma once

#include "core/utilities.h"
#include <memory>

namespace vulkan {
class CommandPool;
class Device;
class DeviceMemory;
class Image;
class ImageView;

class DepthBuffer final {
public:
    DepthBuffer(CommandPool& command_pool, VkExtent2D extent);
    ~DepthBuffer();

    [[nodiscard]] VkFormat format() const { return format_; }
    [[nodiscard]] const ImageView& imageView() const { return *image_view_; }

    static bool hasStencilComponent(const VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

private:

    const VkFormat format_;
    std::unique_ptr<Image> image_;
    std::unique_ptr<DeviceMemory> image_memory_;
    std::unique_ptr<class ImageView> image_view_;
};

}
