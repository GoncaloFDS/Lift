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
                                  const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                                  bool allow_update);
    TopLevelAccelerationStructure(TopLevelAccelerationStructure&& other) noexcept;
    ~TopLevelAccelerationStructure() override = default;

    //    [[nodiscard]] const std::vector<TopLevelInstance>& geometryInstances() const { return geometry_instances_; }

    void generate(VkCommandBuffer command_buffer,
                  Buffer& scratch_buffer,
                  VkDeviceSize scratch_offset,
                  DeviceMemory& result_memory,
                  VkDeviceSize result_offset,
                  Buffer& instance_buffer,
                  DeviceMemory& instance_memory,
                  VkDeviceSize instance_offset,
                  bool update_only) const;

    static VkAccelerationStructureInstanceKHR createInstance(const BottomLevelAccelerationStructure& bottom_level_as,
                                                             const glm::mat4& transform,
                                                             uint32_t instance_id,
                                                             uint32_t hit_group_id);

private:
    std::vector<VkAccelerationStructureInstanceKHR> instances_;
};

}  // namespace vulkan
