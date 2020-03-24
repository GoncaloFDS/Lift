#include "ray_tracing_properties.h"
#include "vulkan/device.h"

namespace vulkan {

RayTracingProperties::RayTracingProperties(const class Device& device) : device_(device) {
    props_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV;

    VkPhysicalDeviceProperties2 props = {};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &props_;
    vkGetPhysicalDeviceProperties2(device.physicalDevice(), &props);
}

}  // namespace vulkan
