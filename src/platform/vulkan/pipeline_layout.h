#pragma once

#include "core/utilities.h"

namespace vulkan {
class DescriptorSetLayout;
class Device;

class PipelineLayout final {
public:
    PipelineLayout(const Device& device, const DescriptorSetLayout& descriptor_set_layout);
    ~PipelineLayout();

private:

    const Device& device_;

VULKAN_HANDLE(VkPipelineLayout, pipelineLayout_)
};

}
