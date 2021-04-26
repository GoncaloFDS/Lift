#pragma once

#include "core/utilities.h"

namespace vulkan {
class Device;

class RayTracingProperties final {
public:
    explicit RayTracingProperties(const Device& device);

    [[nodiscard]] const class Device& device() const { return device_; }

    [[nodiscard]] uint32_t shaderGroupBaseAlignment() const { return pipelineProps_.shaderGroupBaseAlignment; }
    [[nodiscard]] uint32_t shaderGroupHandleSize() const { return pipelineProps_.shaderGroupHandleSize; }

private:
    const class Device& device_;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelProps_{};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR pipelineProps_{};
};
}  // namespace vulkan
