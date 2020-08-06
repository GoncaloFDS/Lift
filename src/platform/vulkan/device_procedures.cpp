
#include "device_procedures.h"
#include "vulkan/device.h"
#include <core.h>

namespace vulkan {

template<class Func>
Func getProcedure(const Device& device, const char* const name) {
    const auto func = reinterpret_cast<Func>(vkGetDeviceProcAddr(device.handle(), name));
    if (func == nullptr) {
        LF_ASSERT(false, "bad procedure");
    }

    return func;
}

DeviceProcedures::DeviceProcedures(const class Device& device) :
    vkCreateAccelerationStructureKHR(getProcedure<PFN_vkCreateAccelerationStructureKHR>(device, "vkCreateAccelerationStructureKHR")),
    vkDestroyAccelerationStructureKHR(getProcedure<PFN_vkDestroyAccelerationStructureKHR>(device, "vkDestroyAccelerationStructureKHR")),
    vkGetAccelerationStructureMemoryRequirementsKHR(getProcedure<PFN_vkGetAccelerationStructureMemoryRequirementsKHR>(device, "vkGetAccelerationStructureMemoryRequirementsKHR")),
    vkBindAccelerationStructureMemoryKHR(getProcedure<PFN_vkBindAccelerationStructureMemoryKHR>(device, "vkBindAccelerationStructureMemoryKHR")),
    vkCmdBuildAccelerationStructureKHR(getProcedure<PFN_vkCmdBuildAccelerationStructureKHR>(device, "vkCmdBuildAccelerationStructureKHR")),
    vkCmdCopyAccelerationStructureKHR(getProcedure<PFN_vkCmdCopyAccelerationStructureKHR>(device, "vkCmdCopyAccelerationStructureKHR")),
    vkCmdTraceRaysKHR(getProcedure<PFN_vkCmdTraceRaysKHR>(device, "vkCmdTraceRaysKHR")),
    vkCreateRayTracingPipelinesKHR(getProcedure<PFN_vkCreateRayTracingPipelinesKHR>(device, "vkCreateRayTracingPipelinesKHR")),
    vkGetRayTracingShaderGroupHandlesKHR(getProcedure<PFN_vkGetRayTracingShaderGroupHandlesKHR>(device, "vkGetRayTracingShaderGroupHandlesKHR")),
    vkGetAccelerationStructureDeviceAddressKHR(getProcedure<PFN_vkGetAccelerationStructureDeviceAddressKHR>(device, "vkGetAccelerationStructureDeviceAddressKHR")),
    vkCmdWriteAccelerationStructuresPropertiesKHR(getProcedure<PFN_vkCmdWriteAccelerationStructuresPropertiesKHR>(device, "vkCmdWriteAccelerationStructuresPropertiesKHR")), device_(device) {
}

DeviceProcedures::~DeviceProcedures() = default;

}  // namespace vulkan
