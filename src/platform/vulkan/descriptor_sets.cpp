
#include "descriptor_sets.h"
#include "descriptor_pool.h"
#include "descriptor_set_layout.h"
#include "device.h"
#include <array>
#include <core.h>

namespace vulkan {

DescriptorSets::DescriptorSets(const DescriptorPool &descriptor_pool,
                               const DescriptorSetLayout &layout,
                               std::map<uint32_t, VkDescriptorType> binding_types,
                               const size_t size) :
    descriptor_pool_(descriptor_pool),
    binding_types_(std::move(binding_types)) {

    std::vector<VkDescriptorSetLayout> layouts(size, layout.handle());

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool.handle();
    alloc_info.descriptorSetCount = static_cast<uint32_t>(size);
    alloc_info.pSetLayouts = layouts.data();

    descriptor_sets_.resize(size);

    vulkanCheck(vkAllocateDescriptorSets(descriptor_pool.device().handle(), &alloc_info, descriptor_sets_.data()),
                "allocate descriptor sets");
}

DescriptorSets::~DescriptorSets() {
    // if (!descriptorSets_.empty())
    //{
    //	vkFreeDescriptorSets(
    //		descriptorPool_.device().handle(),
    //		descriptorPool_.handle(),
    //		static_cast<uint32_t>(descriptorSets_.size()),
    //		descriptorSets_.data());

    //	descriptorSets_.clear();
    //}
}

VkWriteDescriptorSet DescriptorSets::bind(const uint32_t index,
                                          const uint32_t binding,
                                          const VkDescriptorBufferInfo &buffer_info,
                                          const uint32_t count) const {
    VkWriteDescriptorSet descriptor_write = {};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet = descriptor_sets_[index];
    descriptor_write.dstBinding = binding;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorType = getBindingType(binding);
    descriptor_write.descriptorCount = count;
    descriptor_write.pBufferInfo = &buffer_info;

    return descriptor_write;
}

VkWriteDescriptorSet DescriptorSets::bind(const uint32_t index,
                                          const uint32_t binding,
                                          const VkDescriptorImageInfo &image_info,
                                          const uint32_t count) const {
    VkWriteDescriptorSet descriptor_write = {};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet = descriptor_sets_[index];
    descriptor_write.dstBinding = binding;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorType = getBindingType(binding);
    descriptor_write.descriptorCount = count;
    descriptor_write.pImageInfo = &image_info;

    return descriptor_write;
}

VkWriteDescriptorSet DescriptorSets::bind(uint32_t index,
                                          uint32_t binding,
                                          const VkWriteDescriptorSetAccelerationStructureNV &structure_info,
                                          const uint32_t count) const {
    VkWriteDescriptorSet descriptor_write = {};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet = descriptor_sets_[index];
    descriptor_write.dstBinding = binding;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorType = getBindingType(binding);
    descriptor_write.descriptorCount = count;
    descriptor_write.pNext = &structure_info;

    return descriptor_write;
}

void DescriptorSets::updateDescriptors(const std::vector<VkWriteDescriptorSet> &descriptor_writes) {
    vkUpdateDescriptorSets(descriptor_pool_.device().handle(),
                           static_cast<uint32_t>(descriptor_writes.size()),
                           descriptor_writes.data(),
                           0,
                           nullptr);
}

VkDescriptorType DescriptorSets::getBindingType(uint32_t binding) const {
    const auto it = binding_types_.find(binding);
    LF_ASSERT(it != binding_types_.end(), "binding not found");
    return it->second;
}

}  // namespace vulkan
