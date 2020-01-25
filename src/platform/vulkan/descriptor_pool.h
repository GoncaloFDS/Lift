#pragma once

#include "descriptor_binding.h"
#include <vector>

namespace vulkan {
class Device;

class DescriptorPool final {
public:
    DescriptorPool(const Device& device, const std::vector<DescriptorBinding>& descriptor_bindings, size_t max_sets);
    ~DescriptorPool();

    [[nodiscard]] VkDescriptorPool handle() const { return descriptor_pool_; }
    [[nodiscard]] const Device& device() const { return device_; }

private:
    const Device& device_;
    VkDescriptorPool descriptor_pool_{};

};
}
