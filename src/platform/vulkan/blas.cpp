#include "blas.h"
#include "assets/scene.h"
#include "assets/vertex.h"
#include "bottom_level_geometry.h"
#include "device_procedures.h"
#include "vulkan/buffer.h"
#include <core.h>

namespace vulkan {

BottomLevelAccelerationStructure::BottomLevelAccelerationStructure(const class DeviceProcedures& device_procedures,
                                                                   const BottomLevelGeometry& geometries)
    : AccelerationStructure(device_procedures), geometries_(geometries) {

    build_geometry_info_.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_geometry_info_.flags = flags_;
    build_geometry_info_.geometryCount = static_cast<uint32_t>(geometries_.geometry().size());
    build_geometry_info_.pGeometries = geometries_.geometry().data();
    build_geometry_info_.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_geometry_info_.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_geometry_info_.srcAccelerationStructure = nullptr;

    std::vector<uint32_t> maxPrimCount(geometries_.buildOffsetInfo().size());

    for (size_t i = 0; i != maxPrimCount.size(); ++i) {
        maxPrimCount[i] = geometries_.buildOffsetInfo()[i].primitiveCount;
    }

    build_sizes_info_ = getBuildSizes(maxPrimCount.data());
}

BottomLevelAccelerationStructure::BottomLevelAccelerationStructure(BottomLevelAccelerationStructure&& other) noexcept
    : AccelerationStructure(std::move(other)), geometries_(std::move(other.geometries_)) {
}

BottomLevelAccelerationStructure::~BottomLevelAccelerationStructure() {
}

void BottomLevelAccelerationStructure::generate(VkCommandBuffer command_buffer,
                                                Buffer& scratch_buffer,
                                                VkDeviceSize scratch_offset,
                                                Buffer& result_buffer,
                                                const VkDeviceSize result_offset) {

    createAccelerationStructure(result_buffer, result_offset);

    const VkAccelerationStructureBuildRangeInfoKHR* p_build_offset_info = geometries_.buildOffsetInfo().data();

    build_geometry_info_.dstAccelerationStructure = handle();
    build_geometry_info_.scratchData.deviceAddress = scratch_buffer.deviceAddress() + scratch_offset;

    device_procedures_.vkCmdBuildAccelerationStructuresKHR(command_buffer,
                                                           1,
                                                           &build_geometry_info_,
                                                           &p_build_offset_info);
}

}  // namespace vulkan