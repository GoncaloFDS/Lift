#pragma once

#include "acceleration_structure.h"
#include <vector>

namespace assets {
class Procedural;
class Scene;
}

namespace vulkan {

class BottomLevelAccelerationStructure final : public AccelerationStructure {
public:

    BottomLevelAccelerationStructure(const BottomLevelAccelerationStructure&) = delete;
    BottomLevelAccelerationStructure& operator=(const BottomLevelAccelerationStructure&) = delete;
    BottomLevelAccelerationStructure& operator=(BottomLevelAccelerationStructure&&) = delete;

    BottomLevelAccelerationStructure(const class DeviceProcedures& device_procedures,
                                     const std::vector<VkGeometryNV>& geometries,
                                     bool allow_update);
    BottomLevelAccelerationStructure(BottomLevelAccelerationStructure&& other) noexcept;
    ~BottomLevelAccelerationStructure();

    void generate(VkCommandBuffer command_buffer, Buffer& scratch_buffer, VkDeviceSize scratch_offset,
                  DeviceMemory& result_memory, VkDeviceSize result_offset, bool update_only) const;

    static VkGeometryNV createGeometry(const assets::Scene& scene, uint32_t vertex_offset, uint32_t vertex_count,
                                       uint32_t index_offset, uint32_t index_count, bool is_opaque);

    static VkGeometryNV createGeometryAabb(const assets::Scene& scene, uint32_t aabb_offset,
                                           uint32_t aabb_count, bool is_opaque);
private:

    std::vector<VkGeometryNV> geometries_;
};

}
