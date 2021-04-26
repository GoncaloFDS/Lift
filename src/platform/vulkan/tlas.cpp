#include "tlas.h"
#include "blas.h"
#include "device_procedures.h"
#include "vulkan/buffer.h"
#include "vulkan/device.h"
#include <core.h>

namespace vulkan {

TopLevelAccelerationStructure::TopLevelAccelerationStructure(const class DeviceProcedures& device_procedures,
                                                             const VkDeviceAddress instance_address,
                                                             const uint32_t instances_count)
    : AccelerationStructure(device_procedures), instances_count_(instances_count) {
    instances_vk_.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances_vk_.arrayOfPointers = VK_FALSE;
    instances_vk_.data.deviceAddress = instance_address;

    top_as_geometry_.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    top_as_geometry_.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    top_as_geometry_.geometry.instances = instances_vk_;

    build_geometry_info_.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_geometry_info_.flags = flags_;
    build_geometry_info_.geometryCount = 1;
    build_geometry_info_.pGeometries = &top_as_geometry_;
    build_geometry_info_.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_geometry_info_.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build_geometry_info_.srcAccelerationStructure = nullptr;

    build_sizes_info_ = getBuildSizes(&instances_count);
}

TopLevelAccelerationStructure::TopLevelAccelerationStructure(TopLevelAccelerationStructure&& other) noexcept
    : AccelerationStructure(std::move(other)), instances_count_(other.instances_count_) {
}

void TopLevelAccelerationStructure::generate(VkCommandBuffer command_buffer,
                                             Buffer& scratch_buffer,
                                             VkDeviceSize scratch_offset,
                                             Buffer& result_buffer,
                                             const VkDeviceSize result_offset) {
    createAccelerationStructure(result_buffer, result_offset);

    VkAccelerationStructureBuildRangeInfoKHR build_offset_info = {};
    build_offset_info.primitiveCount = instances_count_;

    const VkAccelerationStructureBuildRangeInfoKHR* p_build_offset_info = &build_offset_info;

    build_geometry_info_.dstAccelerationStructure = handle();
    build_geometry_info_.scratchData.deviceAddress = scratch_buffer.deviceAddress() + scratch_offset;

    device_procedures_.vkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &build_geometry_info_, &p_build_offset_info);


}

VkAccelerationStructureInstanceKHR
TopLevelAccelerationStructure::createInstance(const BottomLevelAccelerationStructure& bottom_level_as,
                                              const glm::mat4& transform,
                                              uint32_t instance_id,
                                              uint32_t hit_group_id) {
    const auto& device = bottom_level_as.device();
    const auto& deviceProcedure = bottom_level_as.deviceProcedures();

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = bottom_level_as.handle();

    const VkDeviceAddress address =
        deviceProcedure.vkGetAccelerationStructureDeviceAddressKHR(device.handle(), &addressInfo);

    VkAccelerationStructureInstanceKHR instance = {};
    instance.instanceCustomIndex = instance_id;
    instance.mask = 0xFF;  // The visibility mask is always set of 0xFF, but if some instances would need to be ignored
                           // in some cases, this flag should be passed by the application.
    instance.instanceShaderBindingTableRecordOffset = hit_group_id;
    // Set the hit group index, that will be used to find the shader code to execute when hitting the geometry.
    instance.flags =
        VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;  // Disable culling - more fine control
                                                                       // could be provided by the application
    instance.accelerationStructureReference = address;

    // The instance.transform value only contains 12 values, corresponding to a 4x3 matrix,
    // hence saving the last row that is anyway always (0,0,0,1).
    // Since the matrix is row-major, we simply copy the first 12 values of the original 4x4 matrix
    std::memcpy(&instance.transform, &transform, sizeof(instance.transform));

    return instance;
}

}  // namespace vulkan
