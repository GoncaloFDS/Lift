#pragma once

#include "core/utilities.h"

namespace vulkan {
class Device;

class RayTracingProperties final {
public:

    explicit RayTracingProperties(const Device& device);

    [[nodiscard]] const class Device& device() const { return device_; }

    [[nodiscard]] uint32_t maxDescriptorSetAccelerationStructures() const { return props_.maxDescriptorSetAccelerationStructures; }
    [[nodiscard]] uint64_t maxGeometryCount() const { return props_.maxGeometryCount; }
    [[nodiscard]] uint64_t maxInstanceCount() const { return props_.maxInstanceCount; }
    [[nodiscard]] uint32_t maxRecursionDepth() const { return props_.maxRecursionDepth; }
    [[nodiscard]] uint32_t maxShaderGroupStride() const { return props_.maxShaderGroupStride; }
    [[nodiscard]] uint64_t maxTriangleCount() const { return props_.maxTriangleCount; }
    [[nodiscard]] uint32_t shaderGroupBaseAlignment() const { return props_.shaderGroupBaseAlignment; }
    [[nodiscard]] uint32_t shaderGroupHandleSize() const { return props_.shaderGroupHandleSize; }

private:

    const class Device& device_;
    VkPhysicalDeviceRayTracingPropertiesNV props_{};
};
}
