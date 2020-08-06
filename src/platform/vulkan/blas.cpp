#include "blas.h"
#include "assets/scene.h"
#include "assets/vertex.h"
#include "bottom_level_geometry.h"
#include "device_procedures.h"
#include "vulkan/buffer.h"
#include "vulkan/device.h"
#include <core.h>

namespace vulkan {

BottomLevelAccelerationStructure::BottomLevelAccelerationStructure(const class DeviceProcedures& device_procedures,
                                                                   const BottomLevelGeometry& geometries,
                                                                   const bool allow_update)
    : AccelerationStructure(device_procedures,
                            VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                            geometries.createGeometryTypeInfo(),
                            allow_update),
      geometries_(geometries) {
}

BottomLevelAccelerationStructure::BottomLevelAccelerationStructure(BottomLevelAccelerationStructure&& other) noexcept
    : AccelerationStructure(std::move(other)), geometries_(std::move(other.geometries_)) {
}

BottomLevelAccelerationStructure::~BottomLevelAccelerationStructure() {
}

void BottomLevelAccelerationStructure::generate(VkCommandBuffer command_buffer,
                                                Buffer& scratch_buffer,
                                                VkDeviceSize scratch_offset,
                                                DeviceMemory& result_memory,
                                                VkDeviceSize result_offset,
                                                bool update_only) const {
    LF_ASSERT(!update_only || allow_update_, "[BLAS] cannot update readonly structure")

    const VkAccelerationStructureKHR previous_structure = update_only ? handle() : nullptr;

    // Bind the acceleration structure descriptor to the actual memory that will contain it
    VkBindAccelerationStructureMemoryInfoKHR bind_info = {};
    bind_info.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_KHR;
    bind_info.pNext = nullptr;
    bind_info.accelerationStructure = handle();
    bind_info.memory = result_memory.handle();
    bind_info.memoryOffset = result_offset;
    bind_info.deviceIndexCount = 0;
    bind_info.pDeviceIndices = nullptr;

    vulkanCheck(device_procedures_.vkBindAccelerationStructureMemoryKHR(device().handle(), 1, &bind_info),
                "bind acceleration structure");

    // Build the actual bottom-level acceleration structure
    const auto flags = allow_update_ ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR :
                                       VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    const VkAccelerationStructureGeometryKHR* p_geometry = geometries_.geometry().data();
    const VkAccelerationStructureBuildOffsetInfoKHR* p_build_offset_info = geometries_.buildOffsetInfo().data();

    VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
    build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.pNext = nullptr;
    build_info.flags = flags;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.update = update_only;
    build_info.srcAccelerationStructure = previous_structure;
    build_info.dstAccelerationStructure = handle();
    build_info.geometryArrayOfPointers = false;
    build_info.geometryCount = static_cast<uint32_t>(geometries_.size());
    build_info.ppGeometries = &p_geometry;
    build_info.scratchData.deviceAddress = scratch_buffer.deviceAddress() + scratch_offset;

    device_procedures_.vkCmdBuildAccelerationStructureKHR(command_buffer, 1, &build_info, &p_build_offset_info);
}

}  // namespace vulkan