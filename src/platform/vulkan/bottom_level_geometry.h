#pragma once

#include "core/utilities.h"
#include <vector>

namespace assets {
class Procedural;
class Scene;
}  // namespace assets

namespace vulkan {

class BottomLevelGeometry final {
public:
    [[nodiscard]] size_t size() const { return geometry_.size(); }

    [[nodiscard]] const std::vector<VkAccelerationStructureCreateGeometryTypeInfoKHR>& createGeometryTypeInfo() const {
        return create_geometry_type_info_;
    }
    [[nodiscard]] const std::vector<VkAccelerationStructureGeometryKHR>& geometry() const { return geometry_; }
    [[nodiscard]] const std::vector<VkAccelerationStructureBuildOffsetInfoKHR>& buildOffsetInfo() const { return build_offset_info_; }

    void addGeometryTriangles(const assets::Scene& scene,
                              uint32_t vertex_offset,
                              uint32_t vertex_count,
                              uint32_t index_offset,
                              uint32_t index_count,
                              bool is_opaque);

    void addGeometryAabb(const assets::Scene& scene, uint32_t aabbOffset, uint32_t aabbCount, bool isOpaque);

private:
    // Describe how the acceleration structure is created. It is an indication how large it could be.
    std::vector<VkAccelerationStructureCreateGeometryTypeInfoKHR> create_geometry_type_info_;

    // The geometry to build, addresses of vertices and indices.
    std::vector<VkAccelerationStructureGeometryKHR> geometry_;

    // the number of elements to build and offsets
    std::vector<VkAccelerationStructureBuildOffsetInfoKHR> build_offset_info_;
};

}  // namespace vulkan