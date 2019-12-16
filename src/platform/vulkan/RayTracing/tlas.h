#pragma once

#include "acceleration_structure.h"
#include "Utilities/Glm.hpp"
#include <vector>

namespace vulkan::ray_tracing {
/// Geometry instance, with the layout expected by VK_NV_ray_tracing
struct VkGeometryInstance {
    /// Transform matrix, containing only the top 3 rows
    float transform[12];
    /// Instance index
    uint32_t instanceCustomIndex : 24;
    /// Visibility mask
    uint32_t mask : 8;
    /// Index of the hit group which will be invoked when a ray hits the instance
    uint32_t instanceOffset : 24;
    /// Instance flags, such as culling
    uint32_t flags : 8;
    /// Opaque handle of the bottom-level acceleration structure
    uint64_t accelerationStructureHandle;
};

class BottomLevelAccelerationStructure;

class TopLevelAccelerationStructure final : public AccelerationStructure {
public:

    TopLevelAccelerationStructure(const TopLevelAccelerationStructure&) = delete;
    TopLevelAccelerationStructure& operator=(const TopLevelAccelerationStructure&) = delete;
    TopLevelAccelerationStructure& operator=(TopLevelAccelerationStructure&&) = delete;

    TopLevelAccelerationStructure(const class DeviceProcedures& device_procedures,
                                  const std::vector<VkGeometryInstance>& geometry_instances,
                                  bool allow_update);
    TopLevelAccelerationStructure(TopLevelAccelerationStructure&& other) noexcept;
    ~TopLevelAccelerationStructure() override;

    [[nodiscard]] const std::vector<VkGeometryInstance>& geometryInstances() const { return geometry_instances_; }

    void generate(VkCommandBuffer command_buffer,
                  Buffer& scratch_buffer,
                  VkDeviceSize scratch_offset,
                  DeviceMemory& result_memory,
                  VkDeviceSize result_offset,
                  Buffer& instance_buffer,
                  DeviceMemory& instance_memory,
                  VkDeviceSize instance_offset,
                  bool update_only) const;

    static VkGeometryInstance createGeometryInstance(const BottomLevelAccelerationStructure& bottom_level_as,
                                                     const glm::mat4& transform,
                                                     uint32_t instance_id,
                                                     uint32_t hit_group_index);

private:

    std::vector<VkGeometryInstance> geometry_instances_;
};

}
