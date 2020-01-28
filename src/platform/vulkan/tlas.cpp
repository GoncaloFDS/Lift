#include "tlas.h"
#include "blas.h"
#include "device_procedures.h"
#include "vulkan/buffer.h"
#include "vulkan/device.h"
#include "vulkan/device_memory.h"
#include <cstring>

namespace vulkan {

VkAccelerationStructureCreateInfoNV getCreateInfo(const size_t instance_count, const bool allow_update) {
    const auto flags = allow_update
                       ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV
                       : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;

    VkAccelerationStructureCreateInfoNV structure_info = {};
    structure_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
    structure_info.pNext = nullptr;
    structure_info.info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    structure_info.info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
    structure_info.info.flags = flags;
    structure_info.compactedSize = 0;
    structure_info.info.instanceCount = static_cast<uint32_t>(instance_count);
    structure_info.info.geometryCount = 0; // Since this is a top-level AS, it does not contain any geometry
    structure_info.info.pGeometries = nullptr;

    return structure_info;
}

TopLevelAccelerationStructure::TopLevelAccelerationStructure(
    const class DeviceProcedures& device_procedures,
    const std::vector<VkGeometryInstance>& geometry_instances,
    const bool allow_update) : AccelerationStructure(device_procedures,
                                                     getCreateInfo(geometry_instances.size(), allow_update)),
                               geometry_instances_(geometry_instances) {
}

TopLevelAccelerationStructure::TopLevelAccelerationStructure(TopLevelAccelerationStructure&& other) noexcept :
    AccelerationStructure(std::move(other)),
    geometry_instances_(std::move(other.geometry_instances_)) {
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

    const VkAccelerationStructureNV previousStructure = update_only ? handle() : nullptr;

    // Copy the instance descriptors into the provider buffer.
    const auto instancesBufferSize = geometry_instances_.size() * sizeof(VkGeometryInstance);
    void* data = instance_memory.map(0, instancesBufferSize);
    std::memcpy(data, geometry_instances_.data(), instancesBufferSize);

    // Bind the acceleration structure descriptor to the actual memory that will contain it
    VkBindAccelerationStructureMemoryInfoNV bindInfo = {};
    bindInfo.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
    bindInfo.pNext = nullptr;
    bindInfo.accelerationStructure = handle();
    bindInfo.memory = result_memory.handle();
    bindInfo.memoryOffset = result_offset;
    bindInfo.deviceIndexCount = 0;
    bindInfo.pDeviceIndices = nullptr;

    vulkanCheck(device_procedures_.vkBindAccelerationStructureMemoryNV(device().handle(), 1, &bindInfo),
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
        instance_buffer.handle(),
        instance_offset,
        update_only,
        handle(),
        previousStructure,
        scratch_buffer.handle(),
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
    vulkanCheck(deviceProcedures.vkGetAccelerationStructureHandleNV(device.handle(),
                                                                    bottom_level_as.handle(),
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
