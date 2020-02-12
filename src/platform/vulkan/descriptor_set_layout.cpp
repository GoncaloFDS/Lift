#include "descriptor_set_layout.h"
#include "device.h"

namespace vulkan {

DescriptorSetLayout::DescriptorSetLayout(const Device &device,
                                         const std::vector<DescriptorBinding> &descriptor_bindings) :
    device_(device) {
  std::vector<VkDescriptorSetLayoutBinding> layout_bindings;

  for (const auto &binding : descriptor_bindings) {
    VkDescriptorSetLayoutBinding b = {};
    b.binding = binding.Binding;
    b.descriptorCount = binding.DescriptorCount;
    b.descriptorType = binding.Type;
    b.stageFlags = binding.Stage;

    layout_bindings.push_back(b);
  }

  VkDescriptorSetLayoutCreateInfo layout_info = {};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<uint32_t>(layout_bindings.size());
  layout_info.pBindings = layout_bindings.data();

  vulkanCheck(vkCreateDescriptorSetLayout(device.handle(), &layout_info, nullptr, &layout_),
              "create descriptor set layout");
}

DescriptorSetLayout::~DescriptorSetLayout() {
  if (layout_ != nullptr) {
    vkDestroyDescriptorSetLayout(device_.handle(), layout_, nullptr);
    layout_ = nullptr;
  }
}

}  // namespace vulkan
