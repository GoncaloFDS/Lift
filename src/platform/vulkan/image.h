#pragma once

#include "VulkanError.h"
#include "device_memory.h"

namespace vulkan {
class Buffer;
class CommandPool;
class Device;

class Image final {
public:

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image& operator=(Image&&) = delete;

    Image(const Device& device, VkExtent2D extent, VkFormat format);
    Image(const Device& device, VkExtent2D extent, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage);
    Image(Image&& other) noexcept;
    ~Image();

    [[nodiscard]] const class Device& device() const { return device_; }
    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] VkFormat format() const { return format_; }

    [[nodiscard]] DeviceMemory allocateMemory(VkMemoryPropertyFlags properties) const;
    [[nodiscard]] VkMemoryRequirements getMemoryRequirements() const;

    void transitionImageLayout(CommandPool& command_pool, VkImageLayout new_layout);
    void copyFrom(CommandPool& command_pool, const Buffer& buffer);

private:

    const class Device& device_;
    const VkExtent2D extent_;
    const VkFormat format_;
    VkImageLayout image_layout_;

VULKAN_HANDLE(VkImage, image_)
};

}
