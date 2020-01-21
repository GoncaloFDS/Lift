#include "blas.h"
#include "device_procedures.h"
#include "assets/scene.h"
#include "assets/vertex.h"
#include "platform/vulkan/buffer.h"
#include "platform/vulkan/device.h"
#include "platform/vulkan/single_time_commands.h"

namespace vulkan {

namespace {
VkAccelerationStructureCreateInfoNV GetCreateInfo(const std::vector<VkGeometryNV>& geometries, const bool allowUpdate) {
    const auto flags = allowUpdate
                       ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV
                       : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;

    VkAccelerationStructureCreateInfoNV structureInfo = {};
    structureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
    structureInfo.pNext = nullptr;
    structureInfo.compactedSize = 0;
    structureInfo.info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    structureInfo.info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
    structureInfo.info.flags = flags;
    structureInfo.info.instanceCount = 0; // The bottom-level AS can only contain explicit geometry, and no instances
    structureInfo.info.geometryCount = static_cast<uint32_t>(geometries.size());
    structureInfo.info.pGeometries = geometries.data();

    return structureInfo;
}
}

BottomLevelAccelerationStructure::BottomLevelAccelerationStructure(
    const class DeviceProcedures& device_procedures,
    const std::vector<VkGeometryNV>& geometries,
    const bool allow_update) :
    AccelerationStructure(device_procedures, GetCreateInfo(geometries, allow_update)),
    geometries_(geometries) {
}

BottomLevelAccelerationStructure::BottomLevelAccelerationStructure(BottomLevelAccelerationStructure&& other) noexcept :
    AccelerationStructure(std::move(other)),
    geometries_(std::move(other.geometries_)) {
}

BottomLevelAccelerationStructure::~BottomLevelAccelerationStructure() {
}

void BottomLevelAccelerationStructure::generate(
    VkCommandBuffer command_buffer,
    Buffer& scratch_buffer,
    VkDeviceSize scratch_offset,
    DeviceMemory& result_memory,
    VkDeviceSize result_offset,
    bool update_only) const {
    if (update_only && !allow_update_) {
        throw std::invalid_argument("cannot update readonly structure");
    }

    const VkAccelerationStructureNV previousStructure = update_only ? Handle() : nullptr;

    // Bind the acceleration structure descriptor to the actual memory that will contain it
    VkBindAccelerationStructureMemoryInfoNV bindInfo = {};
    bindInfo.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
    bindInfo.pNext = nullptr;
    bindInfo.accelerationStructure = Handle();
    bindInfo.memory = result_memory.Handle();
    bindInfo.memoryOffset = result_offset;
    bindInfo.deviceIndexCount = 0;
    bindInfo.pDeviceIndices = nullptr;

    vulkanCheck(device_procedures_.vkBindAccelerationStructureMemoryNV(device().Handle(), 1, &bindInfo),
                "bind acceleration structure");

    // Build the actual bottom-level acceleration structure
    const auto flags = allow_update_
                       ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV
                       : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;

    VkAccelerationStructureInfoNV buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    buildInfo.pNext = nullptr;
    buildInfo.flags = flags;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
    buildInfo.instanceCount = 0;
    buildInfo.geometryCount = static_cast<uint32_t>(geometries_.size());
    buildInfo.pGeometries = geometries_.data();

    device_procedures_.vkCmdBuildAccelerationStructureNV(
        command_buffer,
        &buildInfo,
        nullptr,
        0,
        update_only,
        Handle(),
        previousStructure,
        scratch_buffer.Handle(),
        scratch_offset);
}

VkGeometryNV BottomLevelAccelerationStructure::createGeometry(
    const assets::Scene& scene,
    uint32_t vertex_offset, uint32_t vertex_count,
    uint32_t index_offset, uint32_t index_count,
    bool is_opaque) {
    VkGeometryNV geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
    geometry.pNext = nullptr;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
    geometry.geometry.triangles.pNext = nullptr;
    geometry.geometry.triangles.vertexData = scene.VertexBuffer().Handle();
    geometry.geometry.triangles.vertexOffset = vertex_offset;
    geometry.geometry.triangles.vertexCount = vertex_count;
    geometry.geometry.triangles.vertexStride = sizeof(assets::Vertex);
    geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.indexData = scene.IndexBuffer().Handle();
    geometry.geometry.triangles.indexOffset = index_offset;
    geometry.geometry.triangles.indexCount = index_count;
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometry.geometry.triangles.transformData = nullptr;
    geometry.geometry.triangles.transformOffset = 0;
    geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV;
    geometry.flags = is_opaque ? VK_GEOMETRY_OPAQUE_BIT_NV : 0;

    return geometry;
}

VkGeometryNV BottomLevelAccelerationStructure::createGeometryAabb(
    const assets::Scene& scene,
    uint32_t aabb_offset,
    uint32_t aabb_count,
    bool is_opaque) {
    VkGeometryNV geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
    geometry.pNext = nullptr;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_NV;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
    geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV;
    geometry.geometry.aabbs.pNext = nullptr;
    geometry.geometry.aabbs.aabbData = scene.AabbBuffer().Handle();
    geometry.geometry.aabbs.numAABBs = aabb_count;
    geometry.geometry.aabbs.stride = sizeof(glm::vec3) * 2;
    geometry.geometry.aabbs.offset = aabb_offset;
    geometry.flags = is_opaque ? VK_GEOMETRY_OPAQUE_BIT_NV : 0;

    return geometry;
}

}
