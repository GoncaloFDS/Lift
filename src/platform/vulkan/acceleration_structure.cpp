#include "acceleration_structure.h"
#include "device_procedures.h"
#include "vulkan/device.h"
#undef MemoryBarrier

namespace vulkan {

AccelerationStructure::AccelerationStructure(
    const class DeviceProcedures& device_procedures,
    const VkAccelerationStructureTypeKHR acceleration_structure_type,
    const std::vector<VkAccelerationStructureCreateGeometryTypeInfoKHR>& geometries,
    const bool allow_update)
    : device_procedures_(device_procedures), allow_update_(allow_update), device_(device_procedures.device()) {
    const auto flags = allow_update ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR :
                                      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    VkAccelerationStructureCreateInfoKHR create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    create_info.pNext = nullptr;
    create_info.compactedSize = 0;
    create_info.type = acceleration_structure_type;
    create_info.flags = flags;
    create_info.maxGeometryCount = static_cast<uint32_t>(geometries.size());
    create_info.pGeometryInfos = geometries.data();
    create_info.deviceAddress = 0;

    vulkanCheck(device_procedures.vkCreateAccelerationStructureKHR(device_.handle(),
                                                                   &create_info,
                                                                   nullptr,
                                                                   &acceleration_structure_),
                "create acceleration structure");
}

AccelerationStructure::AccelerationStructure(AccelerationStructure&& other) noexcept
    : device_procedures_(other.device_procedures_), allow_update_(other.allow_update_), device_(other.device_),
      acceleration_structure_(other.acceleration_structure_) {
    other.acceleration_structure_ = nullptr;
}

AccelerationStructure::~AccelerationStructure() {
    if (acceleration_structure_ != nullptr) {
        device_procedures_.vkDestroyAccelerationStructureKHR(device_.handle(), acceleration_structure_, nullptr);
        acceleration_structure_ = nullptr;
    }
}

AccelerationStructure::MemoryRequirements AccelerationStructure::getMemoryRequirements() const {
    VkAccelerationStructureMemoryRequirementsInfoKHR memory_requirements_info {};
    memory_requirements_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_KHR;
    memory_requirements_info.pNext = nullptr;
    memory_requirements_info.buildType = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR;
    memory_requirements_info.accelerationStructure = acceleration_structure_;

    // If the descriptor already contains the geometry info, so we can directly compute the estimated size and required
    // scratch memory.
    VkMemoryRequirements2 memory_requirements = {};
    memory_requirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    memory_requirements.pNext = nullptr;

    memory_requirements_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_KHR;
    device_procedures_.vkGetAccelerationStructureMemoryRequirementsKHR(device_.handle(),
                                                                       &memory_requirements_info,
                                                                       &memory_requirements);
    const auto resultRequirements = memory_requirements.memoryRequirements;

    memory_requirements_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_KHR;
    device_procedures_.vkGetAccelerationStructureMemoryRequirementsKHR(device_.handle(),
                                                                       &memory_requirements_info,
                                                                       &memory_requirements);
    const auto buildRequirements = memory_requirements.memoryRequirements;

    memory_requirements_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_KHR;
    device_procedures_.vkGetAccelerationStructureMemoryRequirementsKHR(device_.handle(),
                                                                       &memory_requirements_info,
                                                                       &memory_requirements);
    const auto updateRequirements = memory_requirements.memoryRequirements;

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

AccelerationStructure::MemoryRequirements AccelerationStructure::getTotalRequirements(
    const std::vector<AccelerationStructure::MemoryRequirements>& requirements) {
    AccelerationStructure::MemoryRequirements total {};

    for (const auto& req : requirements) {
        total.result.size += req.result.size;
        total.build.size += req.build.size;
        total.update.size += req.update.size;
    }

    return total;
}

}  // namespace vulkan
