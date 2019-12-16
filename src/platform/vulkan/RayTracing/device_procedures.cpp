#include "device_procedures.h"
#include "platform/vulkan/device.h"
#include <string>

namespace vulkan::ray_tracing {

namespace {
template<class Func>
Func getProcedure(const Device& device, const char* const name) {
    const auto func = reinterpret_cast<Func>(vkGetDeviceProcAddr(device.Handle(), name));
    if (func == nullptr) {
//			Throw(std::runtime_error(std::string("failed to get address of '") + name + "'"));
    }

    return func;
}
}

DeviceProcedures::DeviceProcedures(const class Device& device) :
    vkCreateAccelerationStructureNV(getProcedure<PFN_vkCreateAccelerationStructureNV>(device, "vkCreateAccelerationStructureNV")),
    vkDestroyAccelerationStructureNV(getProcedure<PFN_vkDestroyAccelerationStructureNV>(device, "vkDestroyAccelerationStructureNV")),
    vkGetAccelerationStructureMemoryRequirementsNV(getProcedure<PFN_vkGetAccelerationStructureMemoryRequirementsNV>(device, "vkGetAccelerationStructureMemoryRequirementsNV")),
    vkBindAccelerationStructureMemoryNV(getProcedure<PFN_vkBindAccelerationStructureMemoryNV>(device, "vkBindAccelerationStructureMemoryNV")),
    vkCmdBuildAccelerationStructureNV(getProcedure<PFN_vkCmdBuildAccelerationStructureNV>(device, "vkCmdBuildAccelerationStructureNV")),
    vkCmdCopyAccelerationStructureNV(getProcedure<PFN_vkCmdCopyAccelerationStructureNV>(device, "vkCmdCopyAccelerationStructureNV")),
    vkCmdTraceRaysNV(getProcedure<PFN_vkCmdTraceRaysNV>(device, "vkCmdTraceRaysNV")),
    vkCreateRayTracingPipelinesNV(getProcedure<PFN_vkCreateRayTracingPipelinesNV>(device, "vkCreateRayTracingPipelinesNV")),
    vkGetRayTracingShaderGroupHandlesNV(getProcedure<PFN_vkGetRayTracingShaderGroupHandlesNV>(device, "vkGetRayTracingShaderGroupHandlesNV")),
    vkGetAccelerationStructureHandleNV(getProcedure<PFN_vkGetAccelerationStructureHandleNV>(device, "vkGetAccelerationStructureHandleNV")),
    vkCmdWriteAccelerationStructuresPropertiesNV(getProcedure<PFN_vkCmdWriteAccelerationStructuresPropertiesNV>(device, "vkCmdWriteAccelerationStructuresPropertiesNV")),
    vkCompileDeferredNV(getProcedure<PFN_vkCompileDeferredNV>(device, "vkCompileDeferredNV")),
    device_(device) {
}

DeviceProcedures::~DeviceProcedures() {
}

}
