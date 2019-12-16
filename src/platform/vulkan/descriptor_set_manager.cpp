#include "DescriptorSetManager.h"
#include "descriptor_pool.h"
#include "descriptor_set_layout.h"
#include "DescriptorSets.h"
#include "Device.h"
#include <memory>
#include <set>
#include <core.h>

namespace Vulkan {

DescriptorSetManager::DescriptorSetManager(const Device& device, const std::vector<DescriptorBinding>& descriptorBindings, const size_t maxSets)
{
	// Sanity check to avoid binding different resources to the same binding point.
	std::map<uint32_t, VkDescriptorType> bindingTypes;

	for (const auto& binding : descriptorBindings)
	{
		if (!bindingTypes.insert(std::make_pair(binding.Binding, binding.Type)).second)
		{
			LF_ASSERT(std::invalid_argument("binding collision"));
		}
	}

	descriptorPool_ = std::make_unique<DescriptorPool>(device, descriptorBindings, maxSets);
	descriptorSetLayout_ = std::make_unique<class DescriptorSetLayout>(device, descriptorBindings);
	descriptorSets_ = std::make_unique<class DescriptorSets>(*descriptorPool_, *descriptorSetLayout_, bindingTypes, maxSets);
}

DescriptorSetManager::~DescriptorSetManager()
{
	descriptorSets_.reset();
	descriptorSetLayout_.reset();
	descriptorPool_.reset();
}

}
