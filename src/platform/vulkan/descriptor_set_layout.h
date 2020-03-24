#pragma once

#include "descriptor_binding.h"
#include <vector>

namespace vulkan {
class Device;

class DescriptorSetLayout final {
public:
    DescriptorSetLayout(const Device& device, const std::vector<DescriptorBinding>& descriptor_bindings);
    ~DescriptorSetLayout();

    [[nodiscard]] VkDescriptorSetLayout handle() const { return layout_; }

private:
    const Device& device_;
    VkDescriptorSetLayout layout_ {};
};
}  // namespace vulkan
