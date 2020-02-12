#pragma once

#include "core/utilities.h"
#include <functional>

namespace vulkan {
class Device;

class DeviceProcedures final {
  public:
  explicit DeviceProcedures(const Device &device);
  ~DeviceProcedures();

  [[nodiscard]] const class Device &device() const { return device_; }

  const std::function<VkResult(VkDevice device, const VkAccelerationStructureCreateInfoNV *p_create_info,
                               const VkAllocationCallbacks *p_allocator,
                               VkAccelerationStructureNV *p_acceleration_structure)>
    vkCreateAccelerationStructureNV;

  const std::function<void(VkDevice device, VkAccelerationStructureNV acceleration_structure,
                           const VkAllocationCallbacks *p_allocator)>
    vkDestroyAccelerationStructureNV;

  const std::function<void(VkDevice device, const VkAccelerationStructureMemoryRequirementsInfoNV *p_info,
                           VkMemoryRequirements2KHR *p_memory_requirements)>
    vkGetAccelerationStructureMemoryRequirementsNV;

  const std::function<VkResult(VkDevice device, uint32_t bind_info_count,
                               const VkBindAccelerationStructureMemoryInfoNV *p_bind_infos)>
    vkBindAccelerationStructureMemoryNV;

  const std::function<void(VkCommandBuffer command_buffer, const VkAccelerationStructureInfoNV *p_info,
                           VkBuffer instance_data, VkDeviceSize instance_offset, VkBool32 update,
                           VkAccelerationStructureNV dst, VkAccelerationStructureNV src, VkBuffer scratch,
                           VkDeviceSize scratch_offset)>
    vkCmdBuildAccelerationStructureNV;

  const std::function<void(VkCommandBuffer command_buffer, VkAccelerationStructureNV dst, VkAccelerationStructureNV src,
                           VkCopyAccelerationStructureModeNV mode)>
    vkCmdCopyAccelerationStructureNV;

  const std::function<void(VkCommandBuffer command_buffer, VkBuffer raygen_shader_binding_table_buffer,
                           VkDeviceSize raygen_shader_binding_offset, VkBuffer miss_shader_binding_table_buffer,
                           VkDeviceSize miss_shader_binding_offset, VkDeviceSize miss_shader_binding_stride,
                           VkBuffer hit_shader_binding_table_buffer, VkDeviceSize hit_shader_binding_offset,
                           VkDeviceSize hit_shader_binding_stride, VkBuffer callable_shader_binding_table_buffer,
                           VkDeviceSize callable_shader_binding_offset, VkDeviceSize callable_shader_binding_stride,
                           uint32_t width, uint32_t height, uint32_t depth)>
    vkCmdTraceRaysNV;

  const std::function<VkResult(VkDevice device, VkPipelineCache pipeline_cache, uint32_t create_info_count,
                               const VkRayTracingPipelineCreateInfoNV *p_create_infos,
                               const VkAllocationCallbacks *p_allocator, VkPipeline *p_pipelines)>
    vkCreateRayTracingPipelinesNV;

  const std::function<VkResult(VkDevice device, VkPipeline pipeline, uint32_t first_group, uint32_t group_count,
                               size_t data_size, void *p_data)>
    vkGetRayTracingShaderGroupHandlesNV;

  const std::function<VkResult(VkDevice device, VkAccelerationStructureNV acceleration_structure, size_t data_size,
                               void *p_data)>
    vkGetAccelerationStructureHandleNV;

  const std::function<void(VkCommandBuffer command_buffer, uint32_t acceleration_structure_count,
                           const VkAccelerationStructureNV *p_acceleration_structures, VkQueryType query_type,
                           VkQueryPool query_pool, uint32_t first_query)>
    vkCmdWriteAccelerationStructuresPropertiesNV;

  const std::function<VkResult(VkDevice device, VkPipeline pipeline, uint32_t shader)> vkCompileDeferredNV;

  private:
  const class Device &device_;
};
}  // namespace vulkan
