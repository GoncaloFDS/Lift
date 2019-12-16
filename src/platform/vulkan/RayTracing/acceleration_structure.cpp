#include "acceleration_structure.h"
#include "device_procedures.h"
#include "platform/vulkan/device.h"
#undef MemoryBarrier

namespace vulkan::ray_tracing {

AccelerationStructure::AccelerationStructure(const class DeviceProcedures& device_procedures,
                                             const VkAccelerationStructureCreateInfoNV& create_info) :
    device_procedures_(device_procedures),
    allow_update_(create_info.info.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV),
    device_(device_procedures.device()) {
    vulkanCheck(device_procedures.vkCreateAccelerationStructureNV(device_.Handle(),
                                                                  &create_info,
                                                                  nullptr,
                                                                  &accelerationStructure_),
                "create acceleration structure");
}

AccelerationStructure::AccelerationStructure(AccelerationStructure&& other) noexcept :
    device_procedures_(other.device_procedures_),
    allow_update_(other.allow_update_),
    device_(other.device_),
    accelerationStructure_(other.accelerationStructure_) {
    other.accelerationStructure_ = nullptr;
}

AccelerationStructure::~AccelerationStructure() {
    if (accelerationStructure_ != nullptr) {
        device_procedures_.vkDestroyAccelerationStructureNV(device_.Handle(), accelerationStructure_, nullptr);
        accelerationStructure_ = nullptr;
    }
}

AccelerationStructure::MemoryRequirements AccelerationStructure::getMemoryRequirements() const {
    VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo{};
    memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
    memoryRequirementsInfo.pNext = nullptr;
    memoryRequirementsInfo.accelerationStructure = accelerationStructure_;

    // If the descriptor already contains the geometry info, so we can directly compute the estimated size and required scratch memory.
    VkMemoryRequirements2 memoryRequirements = {};
    memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
    device_procedures_.vkGetAccelerationStructureMemoryRequirementsNV(device_.Handle(),
                                                                      &memoryRequirementsInfo,
                                                                      &memoryRequirements);
    const auto resultRequirements = memoryRequirements.memoryRequirements;

    memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
    device_procedures_.vkGetAccelerationStructureMemoryRequirementsNV(device_.Handle(),
                                                                      &memoryRequirementsInfo,
                                                                      &memoryRequirements);
    const auto buildRequirements = memoryRequirements.memoryRequirements;

    memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV;
    device_procedures_.vkGetAccelerationStructureMemoryRequirementsNV(device_.Handle(),
                                                                      &memoryRequirementsInfo,
                                                                      &memoryRequirements);
    const auto updateRequirements = memoryRequirements.memoryRequirements;

    return {resultRequirements, buildRequirements, updateRequirements};
}

void AccelerationStructure::memoryBarrier(VkCommandBuffer command_buffer) {
    // Wait for the builder to complete by setting a barrier on the resulting buffer. This is
    // particularly important as the construction of the top-level hierarchy may be called right
    // afterwards, before executing the command list.
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.pNext = nullptr;
    memoryBarrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
    memoryBarrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;

    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
        0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

}