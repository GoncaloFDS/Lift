
#include "device_procedures.h"
#include "vulkan/device.h"
#include <core.h>

namespace vulkan {

template<class Func>
Func getProcedure(const Device &device, const char *const name) {
  const auto func = reinterpret_cast<Func>(vkGetDeviceProcAddr(device.handle(), name));
  if (func == nullptr) { LF_ASSERT(false, "bad procedure"); }

  return func;
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

DeviceProcedures::~DeviceProcedures() = default;

}// namespace vulkan
