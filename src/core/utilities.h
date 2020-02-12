#pragma once

#include <vulkan/vulkan_core.h>

namespace vulkan {

void vulkanCheck(VkResult result, const char* operation);
const char* toString(VkResult result);

}
