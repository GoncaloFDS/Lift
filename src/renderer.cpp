#include "renderer.h"

#include "assets/model.h"
#include "assets/texture.h"
#include "assets/uniform_buffer.h"
#include "core.h"
#include "vulkan/blas.h"
#include "vulkan/depth_buffer.h"
#include "vulkan/extensions_vk.h"
#include "vulkan/fence.h"
#include "vulkan/frame_buffer.h"
#include "vulkan/graphics_pipeline.h"
#include "vulkan/image_memory_barrier.h"
#include "vulkan/instance.h"
#include "vulkan/pipeline_layout.h"
#include "vulkan/ray_tracing_pipeline.h"
#include "vulkan/render_pass.h"
#include "vulkan/semaphore.h"
#include "vulkan/shader_binding_table.h"
#include "vulkan/single_time_commands.h"
#include "vulkan/surface.h"
#include "vulkan/swap_chain.h"
#include "vulkan/tlas.h"
#include <assets/scene.h>

Renderer::Renderer(const vulkan::Instance& instance) {
    surface_ = std::make_unique<vulkan::Surface>(instance);
}

Renderer::~Renderer() {
    deleteSwapChain();
    deleteAccelerationStructures();
    device_procedures_.reset();
    properties_.reset();
    command_pool_.reset();

    device_.reset();
    surface_.reset();
}

void Renderer::init(VkPhysicalDevice physical_device, assets::Scene& scene, vulkan::Instance& instance) {
    LF_ASSERT(!device_, "physical device has already been set");

    std::vector<const char*> required_extensions;
    required_extensions.insert(required_extensions.end(),
                               {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                                VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
                                VK_KHR_RAY_TRACING_EXTENSION_NAME,
                                VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
                                VK_KHR_MAINTENANCE3_EXTENSION_NAME,
                                VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
                                VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                                VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
                                VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
                                VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
                                VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME});

    // Required device features.
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features = {};
    buffer_device_address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    buffer_device_address_features.pNext = nullptr;
    buffer_device_address_features.bufferDeviceAddress = true;

    VkPhysicalDeviceDescriptorIndexingFeatures indexing_features = {};
    indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    indexing_features.pNext = &buffer_device_address_features;
    indexing_features.runtimeDescriptorArray = true;

    VkPhysicalDeviceFeatures device_features = {};
    device_features.fillModeNonSolid = true;
    device_features.samplerAnisotropy = true;

    VkPhysicalDeviceRayTracingFeaturesKHR ray_tracing_features = {};
    ray_tracing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_FEATURES_KHR;
    ray_tracing_features.pNext = &indexing_features;
    ray_tracing_features.rayTracing = true;

    device_ = std::make_unique<vulkan::Device>(physical_device,
                                               *surface_,
                                               required_extensions,
                                               device_features,
                                               &ray_tracing_features);
    command_pool_ = std::make_unique<vulkan::CommandPool>(*device_, device_->graphicsFamilyIndex(), true);
    properties_ = std::make_unique<vulkan::RayTracingProperties>(*device_);
    device_procedures_ = std::make_unique<vulkan::DeviceProcedures>(*device_);

    load_VK_EXTENSION_SUBSET(instance.handle(), vkGetInstanceProcAddr, device_->handle(), vkGetDeviceProcAddr);

    denoiser_.setup(*device_, 0);
}

void Renderer::createOutputImage() {
    const auto extent = swapChain().extent();
    output_image_ = std::make_unique<vulkan::Image>(device(),
                                                    extent,
                                                    VK_FORMAT_R32G32B32A32_SFLOAT,
                                                    VK_IMAGE_TILING_OPTIMAL,
                                                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    output_image_memory_ =
        std::make_unique<vulkan::DeviceMemory>(output_image_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    output_image_view_ = std::make_unique<vulkan::ImageView>(device(),
                                                             output_image_->handle(),
                                                             VK_FORMAT_R32G32B32A32_SFLOAT,
                                                             VK_IMAGE_ASPECT_COLOR_BIT);

    denoised_image_ = std::make_unique<vulkan::Image>(device(),
                                                      extent,
                                                      VK_FORMAT_R32G32B32A32_SFLOAT,
                                                      VK_IMAGE_TILING_OPTIMAL,
                                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    denoised_image_memory_ =
        std::make_unique<vulkan::DeviceMemory>(denoised_image_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    denoised_image_view_ = std::make_unique<vulkan::ImageView>(device(),
                                                               denoised_image_->handle(),
                                                               VK_FORMAT_R32G32B32A32_SFLOAT,
                                                               VK_IMAGE_ASPECT_COLOR_BIT);
}

void Renderer::beginCommand(assets::Scene& scene, size_t current_frame) {
    const auto no_timout = std::numeric_limits<uint64_t>::max();

    auto& in_flight_fence = in_flight_fences_[current_frame];
    current_image_available_semaphore_ = image_available_semaphores_[current_frame].handle();
    current_render_finished_semaphore_ = render_finished_semaphores_[current_frame].handle();

    in_flight_fence.wait(no_timout);

    auto result = vkAcquireNextImageKHR(device_->handle(),
                                        swap_chain_->handle(),
                                        no_timout,
                                        current_image_available_semaphore_,
                                        nullptr,
                                        &current_image_index_);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain(scene);
        return;
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        LF_ASSERT(false, "failed to acquire next image '{0}'", vulkan::toString(result));
    }

    current_command_buffer_ = command_buffers_->begin(current_image_index_);
}

void Renderer::trace(assets::Scene& scene) {
    traceCommand(current_command_buffer_, current_image_index_, scene);
}

void Renderer::render(ImguiLayer& user_interface) {
    user_interface.render(current_command_buffer_, swapChainFrameBuffer(current_image_index_));
}

size_t Renderer::endCommand(assets::Scene& scene, size_t current_frame, assets::UniformBufferObject& ubo) {
    command_buffers_->end(current_image_index_);
    updateUniformBuffer(current_image_index_, ubo);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkCommandBuffer command_buffers[] = {current_command_buffer_};
    VkSemaphore wait_semaphores[] = {current_image_available_semaphore_};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signal_semaphores[] = {current_render_finished_semaphore_};

    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = command_buffers;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    auto& in_flight_fence = in_flight_fences_[current_frame];
    in_flight_fence.reset();

    vulkan::vulkanCheck(vkQueueSubmit(device_->graphicsQueue(), 1, &submit_info, in_flight_fence.handle()),
                        "submit draw command buffer");

    VkSwapchainKHR swap_chains[] = {swap_chain_->handle()};
    VkPresentInfoKHR present_info = {};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swap_chains;
    present_info.pImageIndices = &current_image_index_;
    present_info.pResults = nullptr;  // Optional

    auto result = vkQueuePresentKHR(device_->presentQueue(), &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain(scene);
        return 0;
    }

    LF_ASSERT(result == VK_SUCCESS, "Failed to present image '{0}'", vulkan::toString(result));

    return (current_frame + 1) % in_flight_fences_.size();
}

void Renderer::traceCommand(VkCommandBuffer command_buffer, uint32_t image_index, assets::Scene& scene) {
    const auto extent = swapChain().extent();

    VkDescriptorSet descriptor_sets[] = {ray_tracing_pipeline_->descriptorSet(image_index)};

    VkImageSubresourceRange subresource_range = {};
    subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = 1;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = 1;

    vulkan::ImageMemoryBarrier::insert(command_buffer,
                                       output_image_->handle(),
                                       subresource_range,
                                       0,
                                       VK_ACCESS_SHADER_WRITE_BIT,
                                       VK_IMAGE_LAYOUT_UNDEFINED,
                                       VK_IMAGE_LAYOUT_GENERAL);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, ray_tracing_pipeline_->handle());
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            ray_tracing_pipeline_->pipelineLayout().handle(),
                            0,
                            1,
                            descriptor_sets,
                            0,
                            nullptr);

    // Describe the shader binding table.
    VkStridedBufferRegionKHR raygen_shader_binding_table = {};
    raygen_shader_binding_table.buffer = shader_binding_table_->buffer().handle();
    raygen_shader_binding_table.offset = shader_binding_table_->rayGenOffset();
    raygen_shader_binding_table.stride = shader_binding_table_->rayGenEntrySize();
    raygen_shader_binding_table.size = shader_binding_table_->rayGenSize();

    VkStridedBufferRegionKHR miss_shader_binding_table = {};
    miss_shader_binding_table.buffer = shader_binding_table_->buffer().handle();
    miss_shader_binding_table.offset = shader_binding_table_->missOffset();
    miss_shader_binding_table.stride = shader_binding_table_->missEntrySize();
    miss_shader_binding_table.size = shader_binding_table_->missSize();

    VkStridedBufferRegionKHR hit_shader_binding_table = {};
    hit_shader_binding_table.buffer = shader_binding_table_->buffer().handle();
    hit_shader_binding_table.offset = shader_binding_table_->hitGroupOffset();
    hit_shader_binding_table.stride = shader_binding_table_->hitGroupEntrySize();
    hit_shader_binding_table.size = shader_binding_table_->hitGroupSize();

    VkStridedBufferRegionKHR callable_shader_binding_table = {};

    device_procedures_->vkCmdTraceRaysKHR(command_buffer,
                                          &raygen_shader_binding_table,
                                          &miss_shader_binding_table,
                                          &hit_shader_binding_table,
                                          &callable_shader_binding_table,
                                          extent.width,
                                          extent.height,
                                          1);
}

void Renderer::display() {
    display(current_command_buffer_, current_image_index_);
}

void Renderer::display(VkCommandBuffer command_buffer, uint32_t image_index) {
    if (is_denoised)
        graphics_pipeline_->updateOutputImage(*denoised_image_view_, image_index);
    else
        graphics_pipeline_->updateOutputImage(*output_image_view_, image_index);

    std::array<VkClearValue, 2> clear_values = {};
    clear_values[0].color = {{0.2f, 0.2f, 0.2f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = graphics_pipeline_->renderPass().handle();
    render_pass_info.framebuffer = swap_chain_framebuffers_[image_index].handle();
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = swap_chain_->extent();
    render_pass_info.clearValueCount = (uint32_t) clear_values.size();
    render_pass_info.pClearValues = clear_values.data();

    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    {
        VkDescriptorSet descriptor_set[] = {graphics_pipeline_->descriptorSet(0)};

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_->handle());
        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                graphics_pipeline_->pipelineLayout().handle(),
                                0,
                                1,
                                descriptor_set,
                                0,
                                nullptr);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);
    }
    vkCmdEndRenderPass(command_buffer);
}

void Renderer::denoiseImage() {
    denoiser_.denoiseImage(*device_, current_command_buffer_, *command_pool_, *output_image_, *denoised_image_);
}

void Renderer::createAccelerationStructures(assets::Scene& scene) {
    LF_INFO("Building acceleration structures...");
    const auto timer = std::chrono::high_resolution_clock::now();

    vulkan::SingleTimeCommands::submit(commandPool(), [this, &scene](VkCommandBuffer command_buffer) {
        createBottomLevelStructures(command_buffer, scene);
        vulkan::AccelerationStructure::memoryBarrier(command_buffer);
        createTopLevelStructures(command_buffer, scene);
    });

    top_scratch_buffer_.reset();
    top_scratch_buffer_memory_.reset();
    bottom_scratch_buffer_.reset();
    bottom_scratch_buffer_memory_.reset();

    const auto elapsed =
        std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::high_resolution_clock::now() - timer)
            .count();
    LF_INFO("Built acceleration structures in {0}s", elapsed);
}

void Renderer::deleteAccelerationStructures() {
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

void Renderer::createBottomLevelStructures(VkCommandBuffer command_buffer, assets::Scene& scene) {

    // Bottom level acceleration structure
    // Triangles via vertex buffers. Procedurals via AABBs.
    uint32_t vertex_offset = 0;
    uint32_t index_offset = 0;
    uint32_t aabb_offset = 0;

    std::vector<vulkan::AccelerationStructure::MemoryRequirements> requirements;

    for (const auto& model : scene.models()) {
        const auto vertex_count = static_cast<uint32_t>(model.vertexCount());
        const auto index_count = static_cast<uint32_t>(model.indexCount());

        vulkan::BottomLevelGeometry geometries;
        model.procedural() ?
            geometries.addGeometryAabb(scene, aabb_offset, 1, true) :
            geometries.addGeometryTriangles(scene, vertex_offset, vertex_count, index_offset, index_count, true);

        bottom_as_.emplace_back(*device_procedures_, geometries, false);
        requirements.push_back(bottom_as_.back().getMemoryRequirements());

        vertex_offset += vertex_count * sizeof(assets::Vertex);
        index_offset += index_count * sizeof(uint32_t);
        aabb_offset += sizeof(VkAabbPositionsKHR);
    }

    // Allocate the structure memory.
    const auto total = vulkan::AccelerationStructure::getTotalRequirements(requirements);

    bottom_buffer_ = std::make_unique<vulkan::Buffer>(device(), total.result.size, VK_BUFFER_USAGE_RAY_TRACING_BIT_KHR);
    bottom_buffer_memory_ =
        std::make_unique<vulkan::DeviceMemory>(bottom_buffer_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    bottom_scratch_buffer_ = std::make_unique<vulkan::Buffer>(device(),
                                                              total.build.size,
                                                              VK_BUFFER_USAGE_RAY_TRACING_BIT_KHR |
                                                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    bottom_scratch_buffer_memory_ = std::make_unique<vulkan::DeviceMemory>(
        bottom_scratch_buffer_->allocateMemory(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
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
        result_offset += requirements[i].result.size;
        scratch_offset += requirements[i].build.size;
    }
}

void Renderer::createTopLevelStructures(VkCommandBuffer command_buffer, assets::Scene& scene) {
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    std::vector<vulkan::AccelerationStructure::MemoryRequirements> requirements;

    // Hit group 0: triangles
    // Hit group 1: procedurals
    uint32_t instance_id = 0;

    for (const auto& model : scene.models()) {
        instances.push_back(vulkan::TopLevelAccelerationStructure::createInstance(bottom_as_[instance_id],
                                                                                  glm::mat4(1),
                                                                                  instance_id,
                                                                                  model.procedural() ? 1 : 0));
        instance_id++;
    }

    top_as_.emplace_back(*device_procedures_, instances, false);
    requirements.push_back(top_as_.back().getMemoryRequirements());

    // Allocate the structure memory.
    const auto total = vulkan::AccelerationStructure::getTotalRequirements(requirements);

    top_buffer_ = std::make_unique<vulkan::Buffer>(device(), total.result.size, VK_BUFFER_USAGE_RAY_TRACING_BIT_KHR);
    top_buffer_memory_ =
        std::make_unique<vulkan::DeviceMemory>(top_buffer_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    top_scratch_buffer_ = std::make_unique<vulkan::Buffer>(device(),
                                                           total.build.size,
                                                           VK_BUFFER_USAGE_RAY_TRACING_BIT_KHR |
                                                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    top_scratch_buffer_memory_ = std::make_unique<vulkan::DeviceMemory>(
        top_scratch_buffer_->allocateMemory(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    const size_t instances_buffer_size = sizeof(VkAccelerationStructureInstanceKHR) * instances.size();
    instances_buffer_ = std::make_unique<vulkan::Buffer>(device(),
                                                         instances_buffer_size,
                                                         VK_BUFFER_USAGE_RAY_TRACING_BIT_KHR |
                                                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    instances_buffer_memory_ = std::make_unique<vulkan::DeviceMemory>(
        instances_buffer_->allocateMemory(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
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

void Renderer::createRayTracingPipeline(assets::Scene& scene, Algorithm algorithm) {
    ray_tracing_pipeline_ = std::make_unique<vulkan::RayTracingPipeline>(*device_procedures_,
                                                                         swapChain(),
                                                                         top_as_[0],
                                                                         *output_image_view_,
                                                                         uniformBuffers(),
                                                                         scene,
                                                                         algorithm);

    const std::vector<vulkan::ShaderBindingTable::Entry> ray_gen_programs = {
        {ray_tracing_pipeline_->rayGenShaderIndex(), {}}};
    const std::vector<vulkan::ShaderBindingTable::Entry> miss_programs = {
        {ray_tracing_pipeline_->missShaderIndex(), {}},
        {ray_tracing_pipeline_->shadowMissShaderIndex(), {}}};
    const std::vector<vulkan::ShaderBindingTable::Entry> hit_groups = {
        {ray_tracing_pipeline_->triangleHitGroupIndex(), {}},
        {ray_tracing_pipeline_->proceduralHitGroupIndex(), {}}};

    shader_binding_table_ = std::make_unique<vulkan::ShaderBindingTable>(*device_procedures_,
                                                                         *ray_tracing_pipeline_,
                                                                         *properties_,
                                                                         ray_gen_programs,
                                                                         miss_programs,
                                                                         hit_groups);
}

void Renderer::createSwapChain(assets::Scene& scene, Algorithm algorithm) {
    swap_chain_ = std::make_unique<vulkan::SwapChain>(*device_, vsync_);
    depth_buffer_ = std::make_unique<vulkan::DepthBuffer>(*command_pool_, swap_chain_->extent());

    for (size_t i = 0; i != swap_chain_->imageViews().size(); ++i) {
        image_available_semaphores_.emplace_back(*device_);
        render_finished_semaphores_.emplace_back(*device_);
        in_flight_fences_.emplace_back(*device_, true);
        uniform_buffers_.emplace_back(*device_);
    }

    createOutputImage();
    graphics_pipeline_ = std::make_unique<vulkan::GraphicsPipeline>(*swap_chain_, *depth_buffer_);

    for (const auto& image_view : swap_chain_->imageViews()) {
        swap_chain_framebuffers_.emplace_back(*image_view, graphics_pipeline_->renderPass());
    }

    command_buffers_ = std::make_unique<vulkan::CommandBuffers>(*command_pool_,
                                                                static_cast<uint32_t>(swap_chain_framebuffers_.size()));
    createRayTracingPipeline(scene, algorithm);
}

void Renderer::deleteSwapChain() {
    shader_binding_table_.reset();
    ray_tracing_pipeline_.reset();
    output_image_view_.reset();
    output_image_.reset();
    denoised_image_memory_.reset();
    denoised_image_view_.reset();
    denoised_image_.reset();
    output_image_memory_.reset();
    command_buffers_.reset();
    swap_chain_framebuffers_.clear();
    graphics_pipeline_.reset();
    uniform_buffers_.clear();
    in_flight_fences_.clear();
    render_finished_semaphores_.clear();
    image_available_semaphores_.clear();
    depth_buffer_.reset();
    swap_chain_.reset();
}

void Renderer::recreateSwapChain(assets::Scene& scene) {
    device_->waitIdle();
    deleteSwapChain();
    createSwapChain(scene, Algorithm::PT);
}

void Renderer::waitDeviceIdle() {
    device_->waitIdle();
}

void Renderer::updateUniformBuffer(const uint32_t image_index, assets::UniformBufferObject ubo) {
    uniform_buffers_[image_index].setValue(ubo);
}

void Renderer::setupDenoiser() {
}