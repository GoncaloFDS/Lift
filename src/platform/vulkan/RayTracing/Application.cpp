#include "application.h"
#include "blas.h"
#include "device_procedures.h"
#include "ray_tracing_pipeline.h"
#include "shader_binding_table.h"
#include "tlas.h"
#include "assets/model.h"
#include "assets/scene.h"
#include "core/glm.h"
#include "platform/vulkan/buffer.h"
#include "platform/vulkan/image.h"
#include "platform/vulkan/image_memory_barrier.h"
#include "platform/vulkan/image_view.h"
#include "platform/vulkan/pipeline_layout.h"
#include "platform/vulkan/single_time_commands.h"
#include "platform/vulkan/swap_chain.h"
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

namespace vulkan::ray_tracing {

namespace {
AccelerationStructure::MemoryRequirements getTotalRequirements(const std::vector<AccelerationStructure::MemoryRequirements>& requirements) {
    AccelerationStructure::MemoryRequirements total{};

    for (const auto& req : requirements) {
        total.Result.size += req.Result.size;
        total.Build.size += req.Build.size;
        total.Update.size += req.Update.size;
    }

    return total;
}
}

Application::Application(const WindowProperties& window_properties,
                         const bool vsync,
                         const bool enable_validation_layers) :
    vulkan::Application(window_properties, vsync, enable_validation_layers) {
}

Application::~Application() {
    Application::deleteSwapChain();
    deleteAccelerationStructures();

    device_procedures_.reset();
    properties_.reset();
}

void Application::onDeviceSet() {
    properties_ = std::make_unique<RayTracingProperties>(device());
    device_procedures_ = std::make_unique<DeviceProcedures>(device());
}

void Application::createAccelerationStructures() {
    std::cout << "Building acceleration structures..." << std::endl;
    const auto timer = std::chrono::high_resolution_clock::now();

    SingleTimeCommands::submit(commandPool(), [this](VkCommandBuffer command_buffer) {
        createBottomLevelStructures(command_buffer);
        AccelerationStructure::memoryBarrier(command_buffer);
        createTopLevelStructures(command_buffer);
    });

    top_scratch_buffer_.reset();
    top_scratch_buffer_memory_.reset();
    bottom_scratch_buffer_.reset();
    bottom_scratch_buffer_memory_.reset();

    const auto elapsed = std::chrono::duration<float, std::chrono::seconds::period>(
        std::chrono::high_resolution_clock::now() - timer).count();
    std::cout << "Built acceleration structures in " << elapsed << "s" << std::endl;
}

void Application::deleteAccelerationStructures() {
    top_as_.clear();
    instances_buffer_.reset();
    instances_buffer_memory_.reset();
    top_scratch_buffer_.reset();
    top_scratch_buffer_memory_.reset();
    top_buffer_.reset();
    top_buffer_memory_.reset();

    bottom_as_.clear();
    bottom_scratch_buffer_.reset();
    bottom_scratch_buffer_memory_.reset();
    bottom_buffer_.reset();
    bottom_buffer_memory_.reset();
}

void Application::createSwapChain() {
    vulkan::Application::createSwapChain();

    createOutputImage();

    ray_tracing_pipeline_ = std::make_unique<RayTracingPipeline>(*device_procedures_,
                                                                 swapChain(),
                                                                 top_as_[0],
                                                                 *accumulation_image_view_,
                                                                 *output_image_view_,
                                                                 uniformBuffers(),
                                                                 getScene());

    const std::vector<ShaderBindingTable::Entry> ray_gen_programs = {{ray_tracing_pipeline_->rayGenShaderIndex(), {}}};
    const std::vector<ShaderBindingTable::Entry> miss_programs = {{ray_tracing_pipeline_->missShaderIndex(), {}}};
    const std::vector<ShaderBindingTable::Entry> hit_groups = {
        {ray_tracing_pipeline_->triangleHitGroupIndex(), {}},
        {ray_tracing_pipeline_->proceduralHitGroupIndex(), {}}};

    shader_binding_table_ = std::make_unique<ShaderBindingTable>(*device_procedures_,
                                                                 *ray_tracing_pipeline_,
                                                                 *properties_,
                                                                 ray_gen_programs,
                                                                 miss_programs,
                                                                 hit_groups);

    cudaFree(0);
    optixInit();
}

void Application::deleteSwapChain() {
    shader_binding_table_.reset();
    ray_tracing_pipeline_.reset();
    output_image_view_.reset();
    output_image_.reset();
    output_image_memory_.reset();
    accumulation_image_view_.reset();
    accumulation_image_.reset();
    accumulation_image_memory_.reset();

    vulkan::Application::deleteSwapChain();
}

void Application::render(VkCommandBuffer command_buffer, uint32_t image_index) {
    const auto extent = swapChain().extent();

    VkDescriptorSet descriptor_sets[] = {ray_tracing_pipeline_->descriptorSet(image_index)};

    VkImageSubresourceRange subresource_range;
    subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = 1;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = 1;

    ImageMemoryBarrier::insert(command_buffer, accumulation_image_->Handle(), subresource_range, 0,
                               VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    ImageMemoryBarrier::insert(command_buffer, output_image_->Handle(), subresource_range, 0,
                               VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, ray_tracing_pipeline_->Handle());
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_RAY_TRACING_NV,
                            ray_tracing_pipeline_->pipelineLayout().Handle(),
                            0,
                            1,
                            descriptor_sets,
                            0,
                            nullptr);

    device_procedures_->vkCmdTraceRaysNV(command_buffer,
                                         shader_binding_table_->buffer().Handle(),
                                         shader_binding_table_->rayGenOffset(),
                                         shader_binding_table_->buffer().Handle(),
                                         shader_binding_table_->missOffset(),
                                         shader_binding_table_->missEntrySize(),
                                         shader_binding_table_->buffer().Handle(),
                                         shader_binding_table_->hitGroupOffset(),
                                         shader_binding_table_->hitGroupEntrySize(),
                                         nullptr,
                                         0,
                                         0,
                                         extent.width,
                                         extent.height,
                                         1);

    ImageMemoryBarrier::insert(command_buffer,
                               output_image_->Handle(),
                               subresource_range,
                               VK_ACCESS_SHADER_WRITE_BIT,
                               VK_ACCESS_TRANSFER_READ_BIT,
                               VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    ImageMemoryBarrier::insert(command_buffer,
                               swapChain().images()[image_index],
                               subresource_range,
                               0,
                               VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageCopy copy_region;
    copy_region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.srcOffset = {0, 0, 0};
    copy_region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.dstOffset = {0, 0, 0};
    copy_region.extent = {extent.width, extent.height, 1};

    vkCmdCopyImage(command_buffer,
                   output_image_->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   swapChain().images()[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &copy_region);

    ImageMemoryBarrier::insert(command_buffer,
                               swapChain().images()[image_index],
                               subresource_range,
                               VK_ACCESS_TRANSFER_WRITE_BIT,
                               0,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

void Application::createBottomLevelStructures(VkCommandBuffer command_buffer) {
    const auto& scene = getScene();

    // Bottom level acceleration structure
    // Triangles via vertex buffers. Procedurals via AABBs.
    uint32_t vertex_offset = 0;
    uint32_t index_offset = 0;
    uint32_t aabb_offset = 0;

    std::vector<AccelerationStructure::MemoryRequirements> requirements;

    for (const auto& model : scene.Models()) {
        const auto vertex_count = static_cast<uint32_t>(model.NumberOfVertices());
        const auto index_count = static_cast<uint32_t>(model.NumberOfIndices());
        const std::vector<VkGeometryNV> geometries =
            {
                model.Procedural()
                ? BottomLevelAccelerationStructure::createGeometryAabb(scene, aabb_offset, 1, true)
                : BottomLevelAccelerationStructure::createGeometry(scene,
                                                                   vertex_offset,
                                                                   vertex_count,
                                                                   index_offset,
                                                                   index_count,
                                                                   true)
            };

        bottom_as_.emplace_back(*device_procedures_, geometries, false);
        requirements.push_back(bottom_as_.back().getMemoryRequirements());

        vertex_offset += vertex_count * sizeof(assets::Vertex);
        index_offset += index_count * sizeof(uint32_t);
        aabb_offset += sizeof(glm::vec3) * 2;
    }

    // Allocate the structure memory.
    const auto total = getTotalRequirements(requirements);

    bottom_buffer_ = std::make_unique<Buffer>(device(), total.Result.size, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
    bottom_buffer_memory_ =
        std::make_unique<DeviceMemory>(bottom_buffer_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    bottom_scratch_buffer_ = std::make_unique<Buffer>(device(), total.Build.size, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
    bottom_scratch_buffer_memory_ = std::make_unique<DeviceMemory>(bottom_scratch_buffer_->allocateMemory(
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    // Generate the structures.
    VkDeviceSize result_offset = 0;
    VkDeviceSize scratch_offset = 0;

    for (size_t i = 0; i != bottom_as_.size(); ++i) {
        bottom_as_[i].generate(command_buffer,
                               *bottom_scratch_buffer_,
                               scratch_offset,
                               *bottom_buffer_memory_,
                               result_offset,
                               false);
        result_offset += requirements[i].Result.size;
        scratch_offset += requirements[i].Build.size;
    }
}

void Application::createTopLevelStructures(VkCommandBuffer command_buffer) {
    const auto& scene = getScene();

    // Top level acceleration structure
    std::vector<VkGeometryInstance> geometry_instances;
    std::vector<AccelerationStructure::MemoryRequirements> requirements;

    // Hit group 0: triangles
    // Hit group 1: procedurals
    uint32_t instance_id = 0;

    for (const auto& model : scene.Models()) {
        geometry_instances.push_back(TopLevelAccelerationStructure::createGeometryInstance(
            bottom_as_[instance_id], glm::mat4(1), instance_id, model.Procedural() ? 1 : 0));
        instance_id++;
    }

    top_as_.emplace_back(*device_procedures_, geometry_instances, false);
    requirements.push_back(top_as_.back().getMemoryRequirements());

    // Allocate the structure memory.
    const auto total = getTotalRequirements(requirements);

    top_buffer_ = std::make_unique<Buffer>(device(), total.Result.size, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
    top_buffer_memory_ =
        std::make_unique<DeviceMemory>(top_buffer_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    top_scratch_buffer_ = std::make_unique<Buffer>(device(), total.Build.size, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
    top_scratch_buffer_memory_ = std::make_unique<DeviceMemory>(top_scratch_buffer_->allocateMemory(
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    const size_t instances_buffer_size = sizeof(VkGeometryInstance) * geometry_instances.size();
    instances_buffer_ = std::make_unique<Buffer>(device(), instances_buffer_size, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
    instances_buffer_memory_ = std::make_unique<DeviceMemory>(instances_buffer_->allocateMemory(
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));

    // Generate the structures.
    top_as_[0].generate(command_buffer,
                        *top_scratch_buffer_,
                        0,
                        *top_buffer_memory_,
                        0,
                        *instances_buffer_,
                        *instances_buffer_memory_,
                        0,
                        false);
}

void Application::createOutputImage() {
    const auto extent = swapChain().extent();
    const auto format = swapChain().format();
    const auto tiling = VK_IMAGE_TILING_OPTIMAL;

    accumulation_image_ = std::make_unique<Image>(device(),
                                                  extent,
                                                  VK_FORMAT_R32G32B32A32_SFLOAT,
                                                  VK_IMAGE_TILING_OPTIMAL,
                                                  VK_IMAGE_USAGE_STORAGE_BIT);
    accumulation_image_memory_ =
        std::make_unique<DeviceMemory>(accumulation_image_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    accumulation_image_view_ = std::make_unique<ImageView>(device(),
                                                           accumulation_image_->Handle(),
                                                           VK_FORMAT_R32G32B32A32_SFLOAT,
                                                           VK_IMAGE_ASPECT_COLOR_BIT);

    output_image_ = std::make_unique<Image>(device(),
                                            extent,
                                            format,
                                            tiling,
                                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    output_image_memory_ =
        std::make_unique<DeviceMemory>(output_image_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    output_image_view_ =
        std::make_unique<ImageView>(device(), output_image_->Handle(), format, VK_IMAGE_ASPECT_COLOR_BIT);
}

}
