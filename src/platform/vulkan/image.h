#pragma once

#include "VulkanError.h"
#include "device_memory.h"

namespace Vulkan {
class Buffer;
class CommandPool;
class Device;

class Texture final {
public:

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture& operator=(Texture&&) = delete;

    Texture(const Device& device, VkExtent2D extent, VkFormat format);
    Texture(const Device& device, VkExtent2D extent, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage);
    Texture(Texture&& other) noexcept;
    ~Texture();

    const class Device& Device() const { return device_; }
    VkExtent2D Extent() const { return extent_; }
    VkFormat Format() const { return format_; }

    DeviceMemory AllocateMemory(VkMemoryPropertyFlags properties) const;
    VkMemoryRequirements GetMemoryRequirements() const;

    void TransitionImageLayout(CommandPool& commandPool, VkImageLayout newLayout);
    void CopyFrom(CommandPool& commandPool, const Buffer& buffer);

private:

    const class Device& device_;
    const VkExtent2D extent_;
    const VkFormat format_;
    VkImageLayout imageLayout_;

VULKAN_HANDLE(VkImage, image_)
};

}
