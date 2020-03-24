#pragma once

#include "core/utilities.h"

namespace vulkan {
class DescriptorSetLayout;
class Device;

class PipelineLayout final {
public:
    PipelineLayout(const Device& device, const DescriptorSetLayout& descriptor_set_layout);
    ~PipelineLayout();

    [[nodiscard]] VkPipelineLayout handle() const { return pipeline_layout_; }

private:
    const Device& device_;
    VkPipelineLayout pipeline_layout_ {};
};

}  // namespace vulkan
