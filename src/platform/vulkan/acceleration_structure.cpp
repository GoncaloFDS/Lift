#include "acceleration_structure.h"
#include "device_procedures.h"
#include "vulkan/device.h"
#include "vulkan/buffer.h"
#undef MemoryBarrier

namespace vulkan {

AccelerationStructure::AccelerationStructure(const class DeviceProcedures& device_procedures)
    : device_procedures_(device_procedures), flags_(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR),
      device_(device_procedures.device()) {
}

AccelerationStructure::AccelerationStructure(AccelerationStructure&& other) noexcept
    : device_procedures_(other.device_procedures_), flags_(other.flags_),
      build_geometry_info_(other.build_geometry_info_), build_sizes_info_(other.build_sizes_info_),
      device_(other.device_), acceleration_structure_(other.acceleration_structure_) {
    other.acceleration_structure_ = nullptr;
}

AccelerationStructure::~AccelerationStructure() {
    if (acceleration_structure_ != nullptr) {
        device_procedures_.vkDestroyAccelerationStructureKHR(device_.handle(), acceleration_structure_, nullptr);
        acceleration_structure_ = nullptr;
    }
}

VkAccelerationStructureBuildSizesInfoKHR
AccelerationStructure::getBuildSizes(const uint32_t* p_max_primitive_counts) const {
    VkAccelerationStructureBuildSizesInfoKHR sizes_info = {};
    sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    device_procedures_.vkGetAccelerationStructureBuildSizesKHR(device_.handle(),
                                                               VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                               &build_geometry_info_,
                                                               p_max_primitive_counts,
                                                               &sizes_info);

    return sizes_info;
}

void AccelerationStructure::memoryBarrier(VkCommandBuffer command_buffer) {
    // Wait for the builder to complete by setting a barrier on the resulting buffer. This is
    // particularly important as the construction of the top-level hierarchy may be called right
    // afterwards, before executing the command list.
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.pNext = nullptr;
    memoryBarrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    memoryBarrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0,
                         1,
                         &memoryBarrier,
                         0,
                         nullptr,
                         0,
                         nullptr);
}

void AccelerationStructure::createAccelerationStructure(Buffer& result_buffer, VkDeviceSize result_offset) {
    VkAccelerationStructureCreateInfoKHR create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    create_info.pNext = nullptr;
    create_info.type = build_geometry_info_.type;
    create_info.size = buildSizes().accelerationStructureSize;
    create_info.buffer = result_buffer.handle();
    create_info.offset = result_offset;

    device_procedures_.vkCreateAccelerationStructureKHR(device_.handle(), &create_info, nullptr, &acceleration_structure_);
}


}  // namespace vulkan
