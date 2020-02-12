#pragma once

#include "core/utilities.h"
#include "device_memory.h"
#include <vulkan/vulkan.hpp>

namespace vulkan {
class Buffer;
class CommandPool;
class Device;

class Image final {
  public:
  Image(const Image &) = delete;
  Image &operator=(const Image &) = delete;
  Image &operator=(Image &&) = delete;

  Image(const Device &device, VkExtent2D extent, VkFormat format);
  Image(const Device &device, VkExtent2D extent, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage);
  Image(Image &&other) noexcept;
  ~Image();

  [[nodiscard]] VkImage handle() const { return image_; }
  [[nodiscard]] const class Device &device() const { return device_; }
  [[nodiscard]] VkExtent2D extent() const { return extent_; }
  [[nodiscard]] VkFormat format() const { return format_; }

  [[nodiscard]] DeviceMemory allocateMemory(VkMemoryPropertyFlags properties) const;
  [[nodiscard]] VkMemoryRequirements getMemoryRequirements() const;

  //    void transitionImageLayout(CommandPool& command_pool, VkImageLayout new_layout);
  void transitionImageLayout(CommandPool &command_pool, const vk::ImageLayout new_layout);
  void copyFromBuffer(CommandPool &command_pool, const vk::Buffer &buffer);
  void copyToBuffer(CommandPool &command_pool, const vk::Buffer &buffer);

  private:
  vk::AccessFlags accessFlagsForLayout(vk::ImageLayout layout);
  vk::PipelineStageFlags pipelineStageForLayout(vk::ImageLayout layout);

  private:
  const class Device &device_;
  const VkExtent2D extent_;
  const VkFormat format_;
  vk::ImageLayout image_layout_;
  VkImage image_ {};
};

}  // namespace vulkan
