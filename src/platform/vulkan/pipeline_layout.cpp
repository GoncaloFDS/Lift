#include "pipeline_layout.h"
#include "descriptor_set_layout.h"
#include "device.h"

namespace vulkan {

PipelineLayout::PipelineLayout(const Device& device, const DescriptorSetLayout& descriptor_set_layout) :
    device_(device) {
    VkDescriptorSetLayout descriptor_set_layouts[] = {descriptor_set_layout.handle()};

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = descriptor_set_layouts;
    pipeline_layout_info.pushConstantRangeCount = 0; // Optional
    pipeline_layout_info.pPushConstantRanges = nullptr; // Optional

    vulkanCheck(vkCreatePipelineLayout(device_.handle(), &pipeline_layout_info, nullptr, &pipeline_layout_),
                "create pipeline layout");
}

PipelineLayout::~PipelineLayout() {
    if (pipeline_layout_ != nullptr) {
        vkDestroyPipelineLayout(device_.handle(), pipeline_layout_, nullptr);
        pipeline_layout_ = nullptr;
    }
}

}
