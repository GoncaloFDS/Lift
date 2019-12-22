#pragma once

#include "core/utilities.h"

namespace vulkan {
class Device;

class CommandPool final {
public:
    CommandPool(const Device& device, uint32_t queue_family_index, bool allow_reset);
    ~CommandPool();

    [[nodiscard]] const class Device& device() const { return device_; }

private:

    const class Device& device_;

VULKAN_HANDLE(VkCommandPool, commandPool_)
};

}
