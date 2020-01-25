#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#undef APIENTRY

namespace vulkan {

void vulkanCheck(VkResult result, const char* operation);
const char* toString(VkResult result);

}
