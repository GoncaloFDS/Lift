#pragma once

#include "core/utilities.h"
#include <map>
#include <vector>

namespace vulkan {
class Buffer;
class DescriptorPool;
class DescriptorSetLayout;
class ImageView;

class DescriptorSets final {
public:
    DescriptorSets(const DescriptorPool& descriptor_pool,
                   const DescriptorSetLayout& layout,
                   std::map<uint32_t, VkDescriptorType> binding_types,
                   size_t size);

    ~DescriptorSets();

    [[nodiscard]] VkDescriptorSet handle(uint32_t index) const { return descriptor_sets_[index]; }

    [[nodiscard]] VkWriteDescriptorSet
    bind(uint32_t index, uint32_t binding, const VkDescriptorBufferInfo& buffer_info, uint32_t count = 1) const;
    [[nodiscard]] VkWriteDescriptorSet
    bind(uint32_t index, uint32_t binding, const VkDescriptorImageInfo& image_info, uint32_t count = 1) const;
    [[nodiscard]] VkWriteDescriptorSet bind(uint32_t index,
                                            uint32_t binding,
                                            const VkWriteDescriptorSetAccelerationStructureNV& structure_info,
                                            uint32_t count = 1) const;

    void updateDescriptors(const std::vector<VkWriteDescriptorSet>& descriptor_writes);

private:
    [[nodiscard]] VkDescriptorType getBindingType(uint32_t binding) const;

    const DescriptorPool& descriptor_pool_;
    const std::map<uint32_t, VkDescriptorType> binding_types_;

    std::vector<VkDescriptorSet> descriptor_sets_;
};

}  // namespace vulkan