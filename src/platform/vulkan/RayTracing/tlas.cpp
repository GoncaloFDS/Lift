#include "tlas.h"
#include "blas.h"
#include "device_procedures.h"
#include "platform/vulkan/buffer.h"
#include "platform/vulkan/device.h"
#include "platform/vulkan/device_memory.h"
#include <cstring>

namespace vulkan::ray_tracing {

namespace {
VkAccelerationStructureCreateInfoNV GetCreateInfo(const size_t instanceCount, const bool allowUpdate) {
    const auto flags = allowUpdate
                       ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV
                       : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;

    VkAccelerationStructureCreateInfoNV structureInfo = {};
    structureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
    structureInfo.pNext = nullptr;
    structureInfo.info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    structureInfo.info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
    structureInfo.info.flags = flags;
    structureInfo.compactedSize = 0;
    structureInfo.info.instanceCount = static_cast<uint32_t>(instanceCount);
    structureInfo.info.geometryCount = 0; // Since this is a top-level AS, it does not contain any geometry
    structureInfo.info.pGeometries = nullptr;

    return structureInfo;
}
}

TopLevelAccelerationStructure::TopLevelAccelerationStructure(
    const class DeviceProcedures& device_procedures,
    const std::vector<VkGeometryInstance>& geometry_instances,
    const bool allow_update) :
    AccelerationStructure(device_procedures, GetCreateInfo(geometry_instances.size(), allow_update)),
    geometry_instances_(geometry_instances) {
}

TopLevelAccelerationStructure::TopLevelAccelerationStructure(TopLevelAccelerationStructure&& other) noexcept :
    AccelerationStructure(std::move(other)),
    geometry_instances_(std::move(other.geometry_instances_)) {
}

TopLevelAccelerationStructure::~TopLevelAccelerationStructure() {
}

void TopLevelAccelerationStructure::generate(
    VkCommandBuffer command_buffer,
    Buffer& scratch_buffer,
    VkDeviceSize scratch_offset,
    DeviceMemory& result_memory,
    VkDeviceSize result_offset,
    Buffer& instance_buffer,
    DeviceMemory& instance_memory,
    VkDeviceSize instance_offset,
    bool update_only) const {
    if (update_only && !allow_update_) {
//		throw std::invalid_argument("cannot update readonly structure");
    }

    const VkAccelerationStructureNV previousStructure = update_only ? Handle() : nullptr;

    // Copy the instance descriptors into the provider buffer.
    const auto instancesBufferSize = geometry_instances_.size() * sizeof(VkGeometryInstance);
    void* data = instance_memory.map(0, instancesBufferSize);
    std::memcpy(data, geometry_instances_.data(), instancesBufferSize);

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
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
    buildInfo.instanceCount = static_cast<uint32_t>(geometry_instances_.size());
    buildInfo.geometryCount = 0;
    buildInfo.pGeometries = nullptr;

    device_procedures_.vkCmdBuildAccelerationStructureNV(
        command_buffer,
        &buildInfo,
        instance_buffer.Handle(),
        instance_offset,
        update_only,
        Handle(),
        previousStructure,
        scratch_buffer.Handle(),
        scratch_offset);
}

VkGeometryInstance TopLevelAccelerationStructure::createGeometryInstance(
    const BottomLevelAccelerationStructure& bottom_level_as,
    const glm::mat4& transform,
    uint32_t instance_id,
    uint32_t hit_group_index) {
    const auto& device = bottom_level_as.device();
    const auto& deviceProcedures = bottom_level_as.deviceProcedures();

    uint64_t accelerationStructureHandle;
    vulkanCheck(deviceProcedures.vkGetAccelerationStructureHandleNV(device.Handle(),
                                                                    bottom_level_as.Handle(),
                                                                    sizeof(uint64_t),
                                                                    &accelerationStructureHandle),
                "get acceleration structure handle");

    VkGeometryInstance geometryInstance = {};
    std::memcpy(geometryInstance.transform, &transform, sizeof(glm::mat4));
    geometryInstance.instanceCustomIndex = instance_id;
    geometryInstance.mask =
        0xFF; // The visibility mask is always set of 0xFF, but if some instances would need to be ignored in some cases, this flag should be passed by the application.
    geometryInstance.instanceOffset =
        hit_group_index; // Set the hit group index, that will be used to find the shader code to execute when hitting the geometry.
    geometryInstance.flags =
        VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV; // Disable culling - more fine control could be provided by the application
    geometryInstance.accelerationStructureHandle = accelerationStructureHandle;

    return geometryInstance;
}

}
