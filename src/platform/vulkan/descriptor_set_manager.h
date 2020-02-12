#pragma once

#include "descriptor_binding.h"
#include <memory>
#include <vector>

namespace vulkan {
class Device;
class DescriptorPool;
class DescriptorSetLayout;
class DescriptorSets;

class DescriptorSetManager final {
  public:
  explicit DescriptorSetManager(const Device &device, const std::vector<DescriptorBinding> &descriptor_bindings,
                                size_t max_sets);
  ~DescriptorSetManager();

  [[nodiscard]] const DescriptorSetLayout &descriptorSetLayout() const { return *descriptor_set_layout_; }
  class DescriptorSets &descriptorSets() {
    return *descriptor_sets_;
  }

  private:
  std::unique_ptr<DescriptorPool> descriptor_pool_;
  std::unique_ptr<class DescriptorSetLayout> descriptor_set_layout_;
  std::unique_ptr<class DescriptorSets> descriptor_sets_;
};

}  // namespace vulkan
