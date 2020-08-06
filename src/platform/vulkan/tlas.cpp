#include "tlas.h"
#include "blas.h"
#include "device_procedures.h"
#include "vulkan/buffer.h"
#include "vulkan/device.h"
#include <core.h>

namespace vulkan {

std::vector<VkAccelerationStructureCreateGeometryTypeInfoKHR> getCreateGeometryTypeInfo(const size_t instance_count) {
    VkAccelerationStructureCreateGeometryTypeInfoKHR create_geometry_type_info = {};
    create_geometry_type_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_GEOMETRY_TYPE_INFO_KHR;
    create_geometry_type_info.pNext = nullptr;
    create_geometry_type_info.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    create_geometry_type_info.maxPrimitiveCount = static_cast<uint32_t>(instance_count);
    create_geometry_type_info.allowsTransforms = true;

    return std::vector<VkAccelerationStructureCreateGeometryTypeInfoKHR> {create_geometry_type_info};
}

TopLevelAccelerationStructure::TopLevelAccelerationStructure(
    const class DeviceProcedures& device_procedures,
    const std::vector<VkAccelerationStructureInstanceKHR>& instances,
    const bool allow_update)
    : AccelerationStructure(device_procedures,
                            VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                            getCreateGeometryTypeInfo(instances.size()),
                            allow_update),
      instances_(instances) {
}

TopLevelAccelerationStructure::TopLevelAccelerationStructure(TopLevelAccelerationStructure&& other) noexcept
    : AccelerationStructure(std::move(other)), instances_(std::move(other.instances_)) {
}

void TopLevelAccelerationStructure::generate(VkCommandBuffer command_buffer,
                                             Buffer& scratch_buffer,
                                             VkDeviceSize scratch_offset,
                                             DeviceMemory& result_memory,
                                             VkDeviceSize result_offset,
                                             Buffer& instance_buffer,
                                             DeviceMemory& instance_memory,
                                             VkDeviceSize instance_offset,
                                             bool update_only) const {
    LF_ASSERT(!update_only || allow_update_, "[TLAS] cannot update readonly structure")

    const VkAccelerationStructureKHR previous_structure = update_only ? handle() : nullptr;

    // Copy the instance descriptors into the provider buffer.
    const auto instances_buffer_size = instances_.size() * sizeof(VkAccelerationStructureInstanceKHR);
    void* data = instance_memory.map(0, instances_buffer_size);
    std::memcpy(data, instances_.data(), instances_buffer_size);

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

    // Create instance geometry structures
    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.pNext = nullptr;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = false;
    geometry.geometry.instances.data.deviceAddress = instance_buffer.deviceAddress() + instance_offset;

    VkAccelerationStructureBuildOffsetInfoKHR build_offset_info = {};
    build_offset_info.primitiveCount = static_cast<uint32_t>(instances_.size());

    // Build the actual bottom-level acceleration structure
    const auto flags = allow_update_ ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR :
                                       VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    const VkAccelerationStructureGeometryKHR* p_geometry = &geometry;
    const VkAccelerationStructureBuildOffsetInfoKHR* p_build_offset_info = &build_offset_info;

    VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
    build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.pNext = nullptr;
    build_info.flags = flags;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build_info.update = update_only;
    build_info.srcAccelerationStructure = previous_structure;
    build_info.dstAccelerationStructure = handle(), build_info.geometryArrayOfPointers = false;
    build_info.geometryCount = 1;
    build_info.ppGeometries = &p_geometry;
    build_info.scratchData.deviceAddress = scratch_buffer.deviceAddress() + scratch_offset;

    device_procedures_.vkCmdBuildAccelerationStructureKHR(command_buffer, 1, &build_info, &p_build_offset_info);
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
    addressInfo.pNext = nullptr;
    addressInfo.accelerationStructure = bottom_level_as.handle();

    const VkDeviceAddress address =
        deviceProcedure.vkGetAccelerationStructureDeviceAddressKHR(device.handle(), &addressInfo);

    VkAccelerationStructureInstanceKHR instance = {};
    instance.instanceCustomIndex = instance_id;
    instance.mask = 0xFF;  // The visibility mask is always set of 0xFF, but if some instances would need to be ignored
                           // in some cases, this flag should be passed by the application.
    instance.instanceShaderBindingTableRecordOffset = hit_group_id;
    // Set the hit group index, that will be used to find the shader code to execute when hitting the geometry.
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  // Disable culling - more fine control
                                                                                 // could be provided by the application
    instance.accelerationStructureReference = address;

    // The instance.transform value only contains 12 values, corresponding to a 4x3 matrix,
    // hence saving the last row that is anyway always (0,0,0,1).
    // Since the matrix is row-major, we simply copy the first 12 values of the original 4x4 matrix
    std::memcpy(&instance.transform, &transform, sizeof(instance.transform));

    return instance;
}

}  // namespace vulkan
