#pragma once

#include "core/utilities.h"
#include <functional>

namespace vulkan {
class Device;

class DeviceProcedures final {
public:
    explicit DeviceProcedures(const Device& device);
    ~DeviceProcedures();

    [[nodiscard]] const class Device& device() const { return device_; }

    const std::function<VkResult(VkDevice device,
                                 const VkAccelerationStructureCreateInfoKHR* p_create_info,
                                 const VkAllocationCallbacks* p_allocator,
                                 VkAccelerationStructureKHR* p_acceleration_structure)>
        vkCreateAccelerationStructureKHR;

    const std::function<void(VkDevice device,
                             VkAccelerationStructureKHR acceleration_structure,
                             const VkAllocationCallbacks* p_allocator)>
        vkDestroyAccelerationStructureKHR;

    const std::function<void(VkDevice device,
                             VkAccelerationStructureBuildTypeKHR build_type,
                             const VkAccelerationStructureBuildGeometryInfoKHR* p_build_info,
                             const uint32_t* p_max_primitive_counts,
                             VkAccelerationStructureBuildSizesInfoKHR* p_size_info)>
        vkGetAccelerationStructureBuildSizesKHR;

    const std::function<void(VkCommandBuffer command_buffer,
                             uint32_t info_count,
                             const VkAccelerationStructureBuildGeometryInfoKHR* p_infos,
                             const VkAccelerationStructureBuildRangeInfoKHR* const* pp_build_range_infos)>
        vkCmdBuildAccelerationStructuresKHR;

    const std::function<void(VkCommandBuffer command_buffer, const VkCopyAccelerationStructureInfoKHR* mode)>
        vkCmdCopyAccelerationStructureKHR;

    const std::function<void(VkCommandBuffer command_buffer,
                             const VkStridedDeviceAddressRegionKHR* p_raygen_shader_binding_table,
                             const VkStridedDeviceAddressRegionKHR* p_miss_shader_binding_table,
                             const VkStridedDeviceAddressRegionKHR* p_hit_shader_binding_table,
                             const VkStridedDeviceAddressRegionKHR* p_callable_shader_binding_table,
                             uint32_t width,
                             uint32_t height,
                             uint32_t depth)>
        vkCmdTraceRaysKHR;

    const std::function<VkResult(VkDevice device,
                                 VkDeferredOperationKHR deferredOperation,
                                 VkPipelineCache pipeline_cache,
                                 uint32_t create_info_count,
                                 const VkRayTracingPipelineCreateInfoKHR* p_create_infos,
                                 const VkAllocationCallbacks* p_allocator,
                                 VkPipeline* p_pipelines)>
        vkCreateRayTracingPipelinesKHR;

    const std::function<VkResult(VkDevice device,
                                 VkPipeline pipeline,
                                 uint32_t first_group,
                                 uint32_t group_count,
                                 size_t data_size,
                                 void* p_data)>
        vkGetRayTracingShaderGroupHandlesKHR;

    const std::function<VkDeviceAddress(VkDevice device, const VkAccelerationStructureDeviceAddressInfoKHR* p_info)>
        vkGetAccelerationStructureDeviceAddressKHR;

    const std::function<void(VkCommandBuffer command_buffer,
                             uint32_t acceleration_structure_count,
                             const VkAccelerationStructureKHR* p_acceleration_structures,
                             VkQueryType query_type,
                             VkQueryPool query_pool,
                             uint32_t first_query)>
        vkCmdWriteAccelerationStructuresPropertiesKHR;

private:
    const class Device& device_;
};
}  // namespace vulkan
