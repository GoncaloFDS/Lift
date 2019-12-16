#pragma once

#include "platform/vulkan/VulkanError.h"

namespace vulkan {
class Buffer;
class Device;
class DeviceMemory;
}

namespace vulkan::ray_tracing {
class DeviceProcedures;

class AccelerationStructure {
public:

    struct MemoryRequirements {
        VkMemoryRequirements Result;
        VkMemoryRequirements Build;
        VkMemoryRequirements Update;
    };

    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(AccelerationStructure&&) = delete;

    AccelerationStructure(AccelerationStructure&& other) noexcept;
    virtual ~AccelerationStructure();

    const class Device& device() const { return device_; }
    const class DeviceProcedures& deviceProcedures() const { return device_procedures_; }

    MemoryRequirements getMemoryRequirements() const;

    static void memoryBarrier(VkCommandBuffer command_buffer);

protected:

    AccelerationStructure(const class DeviceProcedures& device_procedures,
                          const VkAccelerationStructureCreateInfoNV& create_info);

    const class DeviceProcedures& device_procedures_;
    const bool allow_update_;

private:

    const class Device& device_;

VULKAN_HANDLE(VkAccelerationStructureNV, accelerationStructure_)
};

}
