#pragma once

#define VK_ENABLE_BETA_EXTENSIONS // Until VK_KHR_ray_tracing is out of beta
#include <vulkan/vulkan.h>

namespace vulkan {

void vulkanCheck(VkResult result, const char* operation);
const char* toString(VkResult result);

}  // namespace vulkan
