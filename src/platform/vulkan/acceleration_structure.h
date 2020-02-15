#pragma once

#include "core/utilities.h"
#include <vector>

namespace vulkan {
class Buffer;
class Device;
class DeviceMemory;
}  // namespace vulkan

namespace vulkan {
class DeviceProcedures;

class AccelerationStructure {
public:
    struct MemoryRequirements {
        VkMemoryRequirements result;
        VkMemoryRequirements build;
        VkMemoryRequirements update;
    };

    AccelerationStructure(const AccelerationStructure &) = delete;
    AccelerationStructure &operator=(const AccelerationStructure &) = delete;
    AccelerationStructure &operator=(AccelerationStructure &&) = delete;

    AccelerationStructure(AccelerationStructure &&other) noexcept;
    virtual ~AccelerationStructure();

    [[nodiscard]] VkAccelerationStructureNV handle() const { return acceleration_structure_; }
    [[nodiscard]] const class Device &device() const { return device_; }
    [[nodiscard]] const class DeviceProcedures &deviceProcedures() const { return device_procedures_; }

    [[nodiscard]] MemoryRequirements getMemoryRequirements() const;

    static void memoryBarrier(VkCommandBuffer command_buffer);

    static MemoryRequirements
    getTotalRequirements(const std::vector<AccelerationStructure::MemoryRequirements> &requirements);

protected:
    AccelerationStructure(const class DeviceProcedures &device_procedures,
                          const VkAccelerationStructureCreateInfoNV &create_info);

    const class DeviceProcedures &device_procedures_;
    const bool allow_update_;

private:
    const class Device &device_;

    VkAccelerationStructureNV acceleration_structure_ {};
};

}  // namespace vulkan
