#include "descriptor_set_manager.h"
#include "descriptor_pool.h"
#include "descriptor_set_layout.h"
#include "descriptor_sets.h"
#include "device.h"
#include <memory>
#include <set>
#include <core.h>

namespace vulkan {

DescriptorSetManager::DescriptorSetManager(const Device& device,
                                           const std::vector<DescriptorBinding>& descriptor_bindings,
                                           const size_t max_sets) {
    // Sanity check to avoid binding different resources to the same binding point.
    std::map<uint32_t, VkDescriptorType> binding_types;

    for (const auto& binding : descriptor_bindings) {
        if (!binding_types.insert(std::make_pair(binding.Binding, binding.Type)).second) {
            LF_ASSERT(std::invalid_argument("binding collision"));
        }
    }

    descriptor_pool_ = std::make_unique<DescriptorPool>(device, descriptor_bindings, max_sets);
    descriptor_set_layout_ = std::make_unique<class DescriptorSetLayout>(device, descriptor_bindings);
    descriptor_sets_ =
        std::make_unique<class DescriptorSets>(*descriptor_pool_, *descriptor_set_layout_, binding_types, max_sets);
}

DescriptorSetManager::~DescriptorSetManager() {
    descriptor_sets_.reset();
    descriptor_set_layout_.reset();
    descriptor_pool_.reset();
}

}
