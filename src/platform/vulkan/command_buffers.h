#pragma once

#include "core/utilities.h"
#include <vector>

namespace vulkan {
class CommandPool;

class CommandBuffers final {
  public:
  CommandBuffers(CommandPool &command_pool, uint32_t size);
  ~CommandBuffers();

  [[nodiscard]] uint32_t size() const { return static_cast<uint32_t>(command_buffers_.size()); }
  VkCommandBuffer &operator[](const size_t i) { return command_buffers_[i]; }

  VkCommandBuffer begin(size_t i);
  void end(size_t i);

  private:
  const CommandPool &command_pool_;

  std::vector<VkCommandBuffer> command_buffers_;
};

}  // namespace vulkan
