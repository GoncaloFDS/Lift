#pragma once

#include "descriptor_binding.h"
#include <vector>

namespace vulkan {
class Device;

class DescriptorPool final {
public:
    DescriptorPool(const Device& device, const std::vector<DescriptorBinding>& descriptor_bindings, size_t max_sets);
    ~DescriptorPool();

    [[nodiscard]] const Device& device() const { return device_; }

private:

    const Device& device_;

VULKAN_HANDLE(VkDescriptorPool, descriptorPool_)
};

}
