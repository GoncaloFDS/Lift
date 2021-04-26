#include "bottom_level_geometry.h"
#include "assets/scene.h"
#include "assets/vertex.h"
#include "buffer.h"
#include "device_procedures.h"

namespace vulkan {

void BottomLevelGeometry::addGeometryTriangles(const assets::Scene& scene,
                                               uint32_t vertex_offset,
                                               uint32_t vertex_count,
                                               uint32_t index_offset,
                                               uint32_t index_count,
                                               bool is_opaque) {
    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.pNext = nullptr;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.pNext = nullptr;
    geometry.geometry.triangles.vertexData.deviceAddress = scene.vertexBuffer().deviceAddress();
    geometry.geometry.triangles.vertexStride = sizeof(assets::Vertex);
    geometry.geometry.triangles.maxVertex = vertex_count;
    geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.indexData.deviceAddress = scene.indexBuffer().deviceAddress();
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometry.geometry.triangles.transformData = {};
    geometry.flags = is_opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR ;

    VkAccelerationStructureBuildRangeInfoKHR build_offset_info = {};
    build_offset_info.firstVertex = vertex_offset / sizeof(assets::Vertex);
    build_offset_info.primitiveOffset = index_offset;
    build_offset_info.primitiveCount = index_count / 3;
    build_offset_info.transformOffset = 0;

    geometry_.emplace_back(geometry);
    build_offset_info_.emplace_back(build_offset_info);
}

void BottomLevelGeometry::addGeometryAabb(const assets::Scene& scene,
                                          uint32_t aabbOffset,
                                          uint32_t aabbCount,
                                          bool isOpaque) {
    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.pNext = nullptr;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.geometry.aabbs.pNext = nullptr;
    geometry.geometry.aabbs.data.deviceAddress = scene.aabbBuffer().deviceAddress();
    geometry.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);
    geometry.flags = isOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR ;

    VkAccelerationStructureBuildRangeInfoKHR build_offset_info = {};
    build_offset_info.firstVertex = 0;
    build_offset_info.primitiveOffset = aabbOffset;
    build_offset_info.primitiveCount = aabbCount;
    build_offset_info.transformOffset = 0;

    geometry_.emplace_back(geometry);
    build_offset_info_.emplace_back(build_offset_info);
}

}  // namespace vulkan