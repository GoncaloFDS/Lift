#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#undef APIENTRY

#define VULKAN_HANDLE(VulkanHandleType, name) \
public: \
    VulkanHandleType handle() const { return name; } \
private: \
    VulkanHandleType name{};

namespace vulkan {

void vulkanCheck(VkResult result, const char* operation);
const char* toString(VkResult result);

}
