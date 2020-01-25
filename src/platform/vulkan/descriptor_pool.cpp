#include "descriptor_pool.h"
#include "device.h"

namespace vulkan {

DescriptorPool::DescriptorPool(const vulkan::Device& device,
                               const std::vector<DescriptorBinding>& descriptor_bindings,
                               const size_t max_sets) :
    device_(device) {
    std::vector<VkDescriptorPoolSize> pool_sizes;

    pool_sizes.reserve(descriptor_bindings.size());
    for (const auto& binding : descriptor_bindings) {
        pool_sizes.push_back(VkDescriptorPoolSize{binding.Type,
                                                  static_cast<uint32_t>(binding.DescriptorCount * max_sets )});
    }

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = static_cast<uint32_t>(max_sets);

    vulkanCheck(vkCreateDescriptorPool(device.handle(), &pool_info, nullptr, &descriptorPool_),
                "create descriptor pool");
}

DescriptorPool::~DescriptorPool() {
    if (descriptorPool_ != nullptr) {
        vkDestroyDescriptorPool(device_.handle(), descriptorPool_, nullptr);
        descriptorPool_ = nullptr;
    }
}

}
