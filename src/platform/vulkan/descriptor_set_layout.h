#pragma once

#include "descriptor_binding.h"
#include <vector>

namespace Vulkan
{
	class Device;

	class DescriptorSetLayout final
	{
	public:

		VULKAN_NON_COPIABLE(DescriptorSetLayout)

		DescriptorSetLayout(const Device& device, const std::vector<DescriptorBinding>& descriptorBindings);
		~DescriptorSetLayout();

	private:

		const Device& device_;

		VULKAN_HANDLE(VkDescriptorSetLayout, layout_)
	};

}
