#include <pch.h>
#include "vulkan/window.h"
#include "vulkan/swap_chain.h"
#include "vulkan/device.h"
#include "assets/texture.h"
#include "user_settings.h"
#include "imgui/imgui_layer.h"
#include "application.h"
#include "platform/vulkan/single_time_commands.h"
#include "platform/vulkan/pipeline_layout.h"
#include "platform/vulkan/image_view.h"
#include "platform/vulkan/image_memory_barrier.h"
#include "platform/vulkan/image.h"
#include "platform/vulkan/buffer.h"
#include "platform/vulkan/tlas.h"
#include "platform/vulkan/shader_binding_table.h"
#include "platform/vulkan/ray_tracing_pipeline.h"
#include "platform/vulkan/device_procedures.h"
#include "platform/vulkan/blas.h"
#include "platform/vulkan/depth_buffer.h"
#include "platform/vulkan/fence.h"
#include "platform/vulkan/graphics_pipeline.h"
#include "platform/vulkan/instance.h"
#include "platform/vulkan/render_pass.h"
#include "platform/vulkan/semaphore.h"
#include "platform/vulkan/surface.h"
#include "assets/model.h"
#include "assets/scene.h"
#include "assets/uniform_buffer.h"
#include <events/application_event.h>
#include <events/mouse_event.h>
#include <events/key_event.h>
#include "platform/vulkan/ray_tracing_properties.h"
#include "core/input.h"

void vulkan::Application::timeRender(VkCommandBuffer command_buffer, uint32_t image_index) {
    // Record delta time between calls to Render.
    const auto prev_time = time_;
    time_ = window().time();
    const auto delta_time = time_ - prev_time;

    // Check the current state of the benchmark, update it for the new frame.
    checkAndUpdateBenchmarkState(prev_time);

    // Render the scene
    if (user_settings_.isRayTraced) {
        render(command_buffer, image_index);
    } else {
        rasterize(command_buffer, image_index);
    }

    // Render the UI
    Statistics stats = {};
    stats.framebufferSize = window().framebufferSize();
    stats.frameRate = static_cast<float>(1 / delta_time);

    if (user_settings_.isRayTraced) {
        const auto extent = swapChain().extent();

        stats.rayRate =
            static_cast<float>( double(extent.width * extent.height) * number_of_samples_ / (delta_time * 1000000000));
        stats.totalSamples = total_number_of_samples_;
    }

    user_interface_->render(command_buffer, swapChainFrameBuffer(image_index), stats);
}

void vulkan::Application::loadScene(const uint32_t scene_index) {
    auto[models, textures] = SceneList::allScenes[scene_index].second(camera_initial_sate_);

    LF_WARN("Loading Scene {0}", SceneList::allScenes[scene_index].first.c_str());

    if (textures.empty()) {
        textures.push_back(assets::Texture::loadTexture("../resources/textures/white.png", SamplerConfig()));
    }

    scene_ = std::make_unique<assets::Scene>(commandPool(), std::move(models), std::move(textures), true);
    scene_index_ = scene_index;

    user_settings_.fieldOfView = camera_initial_sate_.fieldOfView;
    user_settings_.aperture = camera_initial_sate_.aperture;
    user_settings_.focusDistance = camera_initial_sate_.focusDistance;
    user_settings_.gammaCorrection = camera_initial_sate_.gammaCorrection;

    camera_x_ = 0;
    camera_y_ = 0;

    period_total_frames_ = 0;
    reset_accumulation_ = true;
}

void vulkan::Application::checkAndUpdateBenchmarkState(double prev_time) {
    if (!user_settings_.benchmark) {
        return;
    }

    // Initialise scene benchmark timers
    if (period_total_frames_ == 0) {
        std::cout << std::endl;
        std::cout << "Benchmark: Start scene #" << scene_index_ << " '" << SceneList::allScenes[scene_index_].first
                  << "'"
                  << std::endl;
        scene_initial_time_ = time_;
        period_initial_time_ = time_;
    }

    // Print out the frame rate at regular intervals.
    {
        const double period = 5;
        const double prev_total_time = prev_time - period_initial_time_;
        const double total_time = time_ - period_initial_time_;

        if (period_total_frames_ != 0
            && static_cast<uint64_t>(prev_total_time / period) != static_cast<uint64_t>(total_time / period)) {
            std::cout << "Benchmark: " << period_total_frames_ / total_time << " fps" << std::endl;
            period_initial_time_ = time_;
            period_total_frames_ = 0;
        }

        period_total_frames_++;
    }

    // If in benchmark mode, bail out from the scene if we've reached the time or sample limit.
    {
        const bool time_limit_reached =
            period_total_frames_ != 0 && window().time() - scene_initial_time_ > user_settings_.benchmarkMaxTime;
        const bool sample_limit_reached = number_of_samples_ == 0;

        if (time_limit_reached || sample_limit_reached) {
            if (!user_settings_.benchmarkNextScenes
                || static_cast<size_t>(user_settings_.sceneIndex) == SceneList::allScenes.size() - 1) {
                window().close();
            }

            std::cout << std::endl;
            user_settings_.sceneIndex += 1;
        }
    }
}

namespace vulkan {

#ifdef NDEBUG
const auto k_validation_layers = std::vector<const char*>();
#else
const auto k_validation_layers = std::vector<const char*>{"VK_LAYER_KHRONOS_validation"};
#endif

Application::Application(const UserSettings& user_settings, const WindowData& window_properties, const bool vsync) :
    user_settings_(user_settings), vsync_(vsync) {
    const auto validation_layers = k_validation_layers;

    window_ = std::make_unique<Window>(window_properties);
    window_->setEventCallbackFn(LF_BIND_EVENT_FN(Application::onEvent));
    instance_ = std::make_unique<Instance>(*window_, validation_layers);
    surface_ = std::make_unique<Surface>(*instance_);

    checkFramebufferSize();
}

Application::~Application() {
    scene_.reset();
    deleteSwapChain();

    deleteAccelerationStructures();
    device_procedures_.reset();
    properties_.reset();
    command_pool_.reset();

    device_.reset();
    surface_.reset();
    instance_.reset();
    window_.reset();
}

assets::UniformBufferObject Application::getUniformBufferObject(VkExtent2D extent) const {
    const auto camera_rot_x = static_cast<float>(camera_y_ / 300.0);
    const auto camera_rot_y = static_cast<float>(camera_x_ / 300.0);

    const auto& init = camera_initial_sate_;
    const auto view = init.modelView;
    const auto model = glm::rotate(mat4(1.0f), camera_rot_y * radians(90.0f), vec3(0.0f, 1.0f, 0.0f)) *
        glm::rotate(mat4(1.0f), camera_rot_x * radians(90.0f), vec3(1.0f, 0.0f, 0.0f));

    assets::UniformBufferObject ubo = {};
    ubo.modelView = view * model;
    ubo.projection = perspective(radians(user_settings_.fieldOfView),
                                 extent.width / static_cast<float>(extent.height),
                                 0.1f,
                                 10000.0f);
    ubo.projection[1][1] *=
        -1; // Inverting Y for vulkan, https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
    ubo.modelViewInverse = glm::inverse(ubo.modelView);
    ubo.projectionInverse = glm::inverse(ubo.projection);
    ubo.aperture = user_settings_.aperture;
    ubo.focusDistance = user_settings_.focusDistance;
    ubo.totalNumberOfSamples = total_number_of_samples_;
    ubo.numberOfSamples = number_of_samples_;
    ubo.numberOfBounces = user_settings_.numberOfBounces;
    ubo.randomSeed = 1;
    ubo.gammaCorrection = user_settings_.gammaCorrection;
    ubo.hasSky = init.hasSky;

    return ubo;
}

void Application::createAccelerationStructures() {
    LF_INFO("Building acceleration structures...");
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
    LF_WARN("Built acceleration structures in {0}s", elapsed);
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

void Application::render(VkCommandBuffer command_buffer, uint32_t image_index) {
    const auto extent = swapChain().extent();

    VkDescriptorSet descriptor_sets[] = {ray_tracing_pipeline_->descriptorSet(image_index)};

    VkImageSubresourceRange subresource_range;
    subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = 1;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = 1;

    ImageMemoryBarrier::insert(command_buffer, accumulation_image_->handle(), subresource_range, 0,
                               VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    ImageMemoryBarrier::insert(command_buffer, output_image_->handle(), subresource_range, 0,
                               VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, ray_tracing_pipeline_->handle());
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_RAY_TRACING_NV,
                            ray_tracing_pipeline_->pipelineLayout().handle(),
                            0,
                            1,
                            descriptor_sets,
                            0,
                            nullptr);

    device_procedures_->vkCmdTraceRaysNV(command_buffer,
                                         shader_binding_table_->buffer().handle(),
                                         shader_binding_table_->rayGenOffset(),
                                         shader_binding_table_->buffer().handle(),
                                         shader_binding_table_->missOffset(),
                                         shader_binding_table_->missEntrySize(),
                                         shader_binding_table_->buffer().handle(),
                                         shader_binding_table_->hitGroupOffset(),
                                         shader_binding_table_->hitGroupEntrySize(),
                                         nullptr,
                                         0,
                                         0,
                                         extent.width,
                                         extent.height,
                                         1);

    ImageMemoryBarrier::insert(command_buffer,
                               output_image_->handle(),
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
                   output_image_->handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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

    for (const auto& model : scene.models()) {
        const auto vertex_count = static_cast<uint32_t>(model.vertexCount());
        const auto index_count = static_cast<uint32_t>(model.indexCount());
        const std::vector<VkGeometryNV> geometries = {
            model.procedural() ? BottomLevelAccelerationStructure::createGeometryAabb(scene, aabb_offset, 1, true)
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
        aabb_offset += sizeof(vec3) * 2;
    }

    // Allocate the structure memory.
    const auto total = AccelerationStructure::getTotalRequirements(requirements);

    bottom_buffer_ = std::make_unique<Buffer>(device(), total.result.size, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
    bottom_buffer_memory_ =
        std::make_unique<DeviceMemory>(bottom_buffer_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    bottom_scratch_buffer_ = std::make_unique<Buffer>(device(), total.build.size, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
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
        result_offset += requirements[i].result.size;
        scratch_offset += requirements[i].build.size;
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

    for (const auto& model : scene.models()) {
        geometry_instances.push_back(TopLevelAccelerationStructure::createGeometryInstance(
            bottom_as_[instance_id], mat4(1), instance_id, model.procedural() ? 1 : 0));
        instance_id++;
    }

    top_as_.emplace_back(*device_procedures_, geometry_instances, false);
    requirements.push_back(top_as_.back().getMemoryRequirements());

    // Allocate the structure memory.
    const auto total = AccelerationStructure::getTotalRequirements(requirements);

    top_buffer_ = std::make_unique<Buffer>(device(), total.result.size, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
    top_buffer_memory_ =
        std::make_unique<DeviceMemory>(top_buffer_->allocateMemory(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

    top_scratch_buffer_ = std::make_unique<Buffer>(device(), total.build.size, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
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

void vulkan::Application::createOutputImage() {
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
                                                           accumulation_image_->handle(),
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
        std::make_unique<ImageView>(device(), output_image_->handle(), format, VK_IMAGE_ASPECT_COLOR_BIT);
}

const std::vector<VkExtensionProperties>& Application::extensions() const {
    return instance_->extensions();
}

const std::vector<VkPhysicalDevice>& Application::physicalDevices() const {
    return instance_->physicalDevices();
}

void Application::setPhysicalDevice(VkPhysicalDevice physical_device) {
    LF_ASSERT(!device_, "physical device has already been set");

    device_ = std::make_unique<Device>(physical_device, *surface_);
    command_pool_ = std::make_unique<CommandPool>(*device_, device_->graphicsFamilyIndex(), true);

    onDeviceSet();
    createSwapChain();
}

void Application::run() {
    LF_ASSERT(device_, "physical device has not been set");

    current_frame_ = 0;

    window_->drawFrame = [this]() {
        prepareFrame();
        drawFrame();
    };
    window_->run();
    device_->waitIdle();
}

void Application::onDeviceSet() {
    properties_ = std::make_unique<RayTracingProperties>(device());
    device_procedures_ = std::make_unique<DeviceProcedures>(device());
    loadScene(user_settings_.sceneIndex);
    createAccelerationStructures();
}

void Application::createSwapChain() {
    while (window_->isMinimized()) {
        window_->waitForEvents();
    }

    swap_chain_ = std::make_unique<SwapChain>(*device_, vsync_);
    depth_buffer_ = std::make_unique<DepthBuffer>(*command_pool_, swap_chain_->extent());

    for (size_t i = 0; i != swap_chain_->imageViews().size(); ++i) {
        image_available_semaphores_.emplace_back(*device_);
        render_finished_semaphores_.emplace_back(*device_);
        in_flight_fences_.emplace_back(*device_, true);
        uniform_buffers_.emplace_back(*device_);
    }

    graphics_pipeline_ = std::make_unique<GraphicsPipeline>(*swap_chain_,
                                                            *depth_buffer_,
                                                            uniform_buffers_,
                                                            getScene(),
                                                            is_wire_frame_);

    for (const auto& image_view : swap_chain_->imageViews()) {
        swap_chain_framebuffers_.emplace_back(*image_view, graphics_pipeline_->renderPass());
    }

    command_buffers_ =
        std::make_unique<CommandBuffers>(*command_pool_, static_cast<uint32_t>(swap_chain_framebuffers_.size()));

    /////
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
    const std::vector<ShaderBindingTable::Entry> hit_groups = {{ray_tracing_pipeline_->triangleHitGroupIndex(), {}},
                                                               {ray_tracing_pipeline_->proceduralHitGroupIndex(), {}}};

    shader_binding_table_ = std::make_unique<ShaderBindingTable>(*device_procedures_,
                                                                 *ray_tracing_pipeline_,
                                                                 *properties_,
                                                                 ray_gen_programs,
                                                                 miss_programs,
                                                                 hit_groups);

    user_interface_ = std::make_unique<ImguiLayer>(commandPool(), swapChain(), depthBuffer(), user_settings_);
    reset_accumulation_ = true;

    checkFramebufferSize();

}

void Application::deleteSwapChain() {
    user_interface_.reset();
    shader_binding_table_.reset();
    ray_tracing_pipeline_.reset();
    output_image_view_.reset();
    output_image_.reset();
    output_image_memory_.reset();
    accumulation_image_view_.reset();
    accumulation_image_.reset();
    accumulation_image_memory_.reset();
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

void vulkan::Application::prepareFrame() {
    // Check if the scene has been changed by the user.
    if (scene_index_ != static_cast<uint32_t>(user_settings_.sceneIndex)) {
        device().waitIdle();
        deleteSwapChain();
        deleteAccelerationStructures();
        loadScene(user_settings_.sceneIndex);
        createAccelerationStructures();
        createSwapChain();
        return;
    }

    // Check if the accumulation buffer needs to be reset.
    if (reset_accumulation_ ||
        user_settings_.requiresAccumulationReset(previous_settings_) ||
        !user_settings_.accumulateRays) {
        total_number_of_samples_ = 0;
        reset_accumulation_ = false;
    }

    previous_settings_ = user_settings_;

    // Keep track of our sample count.
    number_of_samples_ = clamp(user_settings_.maxNumberOfSamples - total_number_of_samples_,
                               0u, user_settings_.numberOfSamples);
    total_number_of_samples_ += number_of_samples_;

}

void Application::drawFrame() {
    const auto no_timeout = std::numeric_limits<uint64_t>::max();

    auto& in_flight_fence = in_flight_fences_[current_frame_];
    const auto image_available_semaphore = image_available_semaphores_[current_frame_].handle();
    const auto render_finished_semaphore = render_finished_semaphores_[current_frame_].handle();

    in_flight_fence.wait(no_timeout);

    uint32_t image_index;
    auto result = vkAcquireNextImageKHR(device_->handle(),
                                        swap_chain_->handle(),
                                        no_timeout,
                                        image_available_semaphore,
                                        nullptr,
                                        &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || is_wire_frame_ != graphics_pipeline_->isWireFrame()) {
        recreateSwapChain();
        return;
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        LF_ASSERT(false, "failed to acquire next image '{0}'", toString(result));
    }

    const auto command_buffer = command_buffers_->begin(image_index);
    timeRender(command_buffer, image_index);
    command_buffers_->end(image_index);

    updateUniformBuffer(image_index);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkCommandBuffer command_buffers[]{command_buffer};
    VkSemaphore wait_semaphores[] = {image_available_semaphore};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signal_semaphores[] = {render_finished_semaphore};

    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = command_buffers;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    in_flight_fence.reset();

    vulkanCheck(vkQueueSubmit(device_->graphicsQueue(), 1, &submit_info, in_flight_fence.handle()),
                "submit draw command buffer");

    VkSwapchainKHR swap_chains[] = {swap_chain_->handle()};
    VkPresentInfoKHR present_info = {};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swap_chains;
    present_info.pImageIndices = &image_index;
    present_info.pResults = nullptr; // Optional

    result = vkQueuePresentKHR(device_->presentQueue(), &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    }

    LF_ASSERT(result == VK_SUCCESS, "Failed to present image '{0}'", toString(result));

    current_frame_ = (current_frame_ + 1) % in_flight_fences_.size();
}

void Application::rasterize(VkCommandBuffer command_buffer, uint32_t image_index) {
    std::array<VkClearValue, 2> clear_values = {};
    clear_values[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = graphics_pipeline_->renderPass().handle();
    render_pass_info.framebuffer = swap_chain_framebuffers_[image_index].handle();
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = swap_chain_->extent();
    render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
    render_pass_info.pClearValues = clear_values.data();

    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    {
        const auto& scene = getScene();

        VkDescriptorSet descriptor_sets[] = {graphics_pipeline_->descriptorSet(image_index)};
        VkBuffer vertex_buffers[] = {scene.vertexBuffer().handle()};
        const VkBuffer index_buffer = scene.indexBuffer().handle();
        VkDeviceSize offsets[] = {0};

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_->handle());
        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                graphics_pipeline_->pipelineLayout().handle(),
                                0,
                                1,
                                descriptor_sets,
                                0,
                                nullptr);
        vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);
        vkCmdBindIndexBuffer(command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);

        uint32_t vertex_offset = 0;
        uint32_t index_offset = 0;

        for (const auto& model : scene.models()) {
            const auto vertex_count = static_cast<uint32_t>(model.vertexCount());
            const auto index_count = static_cast<uint32_t>(model.indexCount());

            vkCmdDrawIndexed(command_buffer, index_count, 1, index_offset, vertex_offset, 0);

            vertex_offset += vertex_count;
            index_offset += index_count;
        }
    }
    vkCmdEndRenderPass(command_buffer);
}

void Application::updateUniformBuffer(const uint32_t image_index) {
    uniform_buffers_[image_index].setValue(getUniformBufferObject(swap_chain_->extent()));
}

void Application::recreateSwapChain() {
    device_->waitIdle();
    deleteSwapChain();
    createSwapChain();
}

void Application::onEvent(Event& event) {
    EventDispatcher dispatcher(event);
    dispatcher.dispatch<WindowCloseEvent>(LF_BIND_EVENT_FN(Application::onWindowClose));
    dispatcher.dispatch<WindowResizeEvent>(LF_BIND_EVENT_FN(Application::onWindowResize));
    dispatcher.dispatch<WindowMinimizeEvent>(LF_BIND_EVENT_FN(Application::onWindowMinimize));

    dispatcher.dispatch<MouseMovedEvent>(LF_BIND_EVENT_FN(Application::onMouseMove));
    dispatcher.dispatch<MouseScrolledEvent>(LF_BIND_EVENT_FN(Application::onMouseScroll));
    dispatcher.dispatch<KeyPressedEvent>(LF_BIND_EVENT_FN(Application::onKeyPress));
    dispatcher.dispatch<KeyReleasedEvent>(LF_BIND_EVENT_FN(Application::onKeyRelease));

}

bool Application::onWindowClose(WindowCloseEvent& e) {
    is_running_ = false;
    LF_INFO(e.toString());
    return false;
}

bool Application::onWindowResize(WindowResizeEvent& e) {
    if (e.height() && e.width()) {
        LF_INFO(e.toString());
    }
    return false;
}

bool Application::onWindowMinimize(WindowMinimizeEvent& e) {
    return false;
}

bool Application::onMouseMove(MouseMovedEvent& e) {
    if (Input::isKeyPressed(LF_MOUSE_BUTTON_1)) {
        camera_x_ += static_cast<float>(e.x() - mouse_x_);
        camera_y_ += static_cast<float>(e.y() - mouse_y_);

        reset_accumulation_ = true;
    }
    mouse_x_ = e.x();
    mouse_y_ = e.y();
    return false;
}

bool Application::onMouseScroll(MouseScrolledEvent& e) {
    return false;
}

bool Application::onKeyPress(KeyPressedEvent& e) {
    if (user_interface_->wantsToCaptureKeyboard()) {
        return true;
    }
    switch (e.keyCode()) {
        case LF_KEY_ESCAPE:
            window().close();
            break;
        case LF_KEY_F1:
            user_settings_.showSettings = !user_settings_.showSettings;
            break;
        case LF_KEY_F2:
            user_settings_.showOverlay = !user_settings_.showOverlay;
            break;
        case LF_KEY_F3:
            user_settings_.isRayTraced = !user_settings_.isRayTraced;
            break;
        default:
            break;

    }
    return false;
}

bool Application::onKeyRelease(KeyReleasedEvent& e) {
    return false;
}

void Application::checkFramebufferSize() const {
    // Check the framebuffer size when requesting a fullscreen window, as it's not guaranteed to match.
    const auto& cfg = window().config();
    const auto fb_size = window().framebufferSize();

    if (user_settings_.benchmark && cfg.fullscreen && (fb_size.width != cfg.width || fb_size.height != cfg.height)) {
        std::ostringstream out;
        out << "framebuffer fullscreen size mismatch (requested: ";
        out << cfg.width << "x" << cfg.height;
        out << ", got: ";
        out << fb_size.width << "x" << fb_size.height << ")";

        LF_ASSERT(false, "framebuffer fullscreen size mismatch (requested: ");
    }
}

}
