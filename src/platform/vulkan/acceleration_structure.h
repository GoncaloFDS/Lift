#pragma once

#include "core/utilities.h"

namespace vulkan {
class Buffer;
class Device;
class DeviceMemory;
}  // namespace vulkan

namespace vulkan {
class DeviceProcedures;

class AccelerationStructure {
public:
    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(AccelerationStructure&&) = delete;

    AccelerationStructure(AccelerationStructure&& other) noexcept;
    virtual ~AccelerationStructure();

    [[nodiscard]] VkAccelerationStructureKHR handle() const { return acceleration_structure_; }
    [[nodiscard]] const class Device& device() const { return device_; }
    [[nodiscard]] const class DeviceProcedures& deviceProcedures() const { return device_procedures_; }
    [[nodiscard]] const VkAccelerationStructureBuildSizesInfoKHR buildSizes() const { return build_sizes_info_; }

    static void memoryBarrier(VkCommandBuffer command_buffer);

protected:
    explicit AccelerationStructure(const class DeviceProcedures& device_procedures);

    VkAccelerationStructureBuildSizesInfoKHR getBuildSizes(const uint32_t* p_max_primitive_counts) const;
    void createAccelerationStructure(Buffer& result_buffer, VkDeviceSize result_offset);

    const class DeviceProcedures& device_procedures_;

    const VkBuildAccelerationStructureFlagsKHR flags_;

    VkAccelerationStructureBuildGeometryInfoKHR build_geometry_info_ {};
    VkAccelerationStructureBuildSizesInfoKHR build_sizes_info_ {};

private:
    const class Device& device_;

    VkAccelerationStructureKHR acceleration_structure_ {};
};

}  // namespace vulkan
