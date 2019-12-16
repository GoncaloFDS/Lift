#pragma once

#include "descriptor_binding.h"
#include <vector>

namespace vulkan {
class Device;

class DescriptorSetLayout final {
public:
    DescriptorSetLayout(const Device& device, const std::vector<DescriptorBinding>& descriptor_bindings);
    ~DescriptorSetLayout();

private:

    const Device& device_;

VULKAN_HANDLE(VkDescriptorSetLayout, layout_)
};

}
