#pragma once

#include "acceleration_structure.h"
#include "core/glm.h"
#include <vector>

namespace vulkan {

class BottomLevelAccelerationStructure;

class TopLevelAccelerationStructure final : public AccelerationStructure {
public:
    TopLevelAccelerationStructure(const TopLevelAccelerationStructure&) = delete;
    TopLevelAccelerationStructure& operator=(const TopLevelAccelerationStructure&) = delete;
    TopLevelAccelerationStructure& operator=(TopLevelAccelerationStructure&&) = delete;

    TopLevelAccelerationStructure(const class DeviceProcedures& device_procedures,
                                  VkDeviceAddress instance_address, uint32_t instances_count);
    TopLevelAccelerationStructure(TopLevelAccelerationStructure&& other) noexcept;
    ~TopLevelAccelerationStructure() override = default;

    //    [[nodiscard]] const std::vector<TopLevelInstance>& geometryInstances() const { return geometry_instances_; }

    void generate(VkCommandBuffer command_buffer,
                  Buffer& scratch_buffer,
                  VkDeviceSize scratch_offset,
                  Buffer& result_buffer,
                  VkDeviceSize result_offset);

    static VkAccelerationStructureInstanceKHR createInstance(const BottomLevelAccelerationStructure& bottom_level_as,
                                                             const glm::mat4& transform,
                                                             uint32_t instance_id,
                                                             uint32_t hit_group_id);

private:
    uint32_t instances_count_;
    VkAccelerationStructureGeometryInstancesDataKHR instances_vk_ {};
    VkAccelerationStructureGeometryKHR top_as_geometry_{};
};

}  // namespace vulkan
