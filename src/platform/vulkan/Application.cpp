#include "Application.h"
#include "buffer.h"
#include "command_pool.h"
#include "command_buffers.h"
#include "depth_buffer.h"
#include "device.h"
#include "fence.h"
#include "frame_buffer.h"
#include "graphics_pipeline.h"
#include "instance.h"
#include "pipeline_layout.h"
#include "render_pass.h"
#include "semaphore.h"
#include "surface.h"
#include "swap_chain.h"
#include "window.h"
#include "assets/model.h"
#include "assets/scene.h"
#include "assets/uniform_buffer.h"
#include <array>
#include <memory>
#include <core.h>

namespace vulkan {

Application::Application(const WindowProperties& window_properties,
                         const bool vsync,
                         const bool enable_validation_layers) :
    vsync_(vsync) {
    const auto validation_layers = enable_validation_layers
                                   ? std::vector<const char*>{"VK_LAYER_KHRONOS_validation"}
                                   : std::vector<const char*>();

    window_ = std::make_unique<class Window>(window_properties);
    instance_ = std::make_unique<Instance>(*window_, validation_layers);
    surface_ = std::make_unique<Surface>(*instance_);
}

Application::~Application() {
    Application::deleteSwapChain();

    command_pool_.reset();
    device_.reset();
    surface_.reset();
    instance_.reset();
    window_.reset();
}

const std::vector<VkExtensionProperties>& Application::extensions() const {
    return instance_->extensions();
}

const std::vector<VkPhysicalDevice>& Application::physicalDevices() const {
    return instance_->physicalDevices();
}

void Application::setPhysicalDevice(VkPhysicalDevice physical_device) {
    if (device_) {
        LF_ASSERT(std::logic_error("physical device has already been set"));
    }

    device_ = std::make_unique<class Device>(physical_device, *surface_);
    command_pool_ = std::make_unique<class CommandPool>(*device_, device_->graphicsFamilyIndex(), true);

    onDeviceSet();

    // Create swap chain and command buffers.
    createSwapChain();
}

void Application::run() {
    if (!device_) {
        LF_ASSERT(std::logic_error("physical device has not been set"));
    }

    current_frame_ = 0;

    window_->drawFrame = [this]() { drawFrame(); };
    window_->onKey = [this](int key, int scan_code, int action, int mods) { onKey(key, scan_code, action, mods); };
    window_->onCursorPosition = [this](double xpos, double ypos) { onCursorPosition(xpos, ypos); };
    window_->onMouseButton = [this](int button, int action, int mods) { onMouseButton(button, action, mods); };
    window_->run();
    device_->waitIdle();
}

void Application::onDeviceSet() {
}

void Application::createSwapChain() {
    // Wait until the window is visible.
    while (window_->isMinimized()) {
        window_->waitForEvents();
    }

    swap_chain_ = std::make_unique<class SwapChain>(*device_, vsync_);
    depth_buffer_ = std::make_unique<class DepthBuffer>(*command_pool_, swap_chain_->extent());

    for (size_t i = 0; i != swap_chain_->imageViews().size(); ++i) {
        image_available_semaphores_.emplace_back(*device_);
        render_finished_semaphores_.emplace_back(*device_);
        in_flight_fences_.emplace_back(*device_, true);
        uniform_buffers_.emplace_back(*device_);
    }

    graphics_pipeline_ = std::make_unique<class GraphicsPipeline>(*swap_chain_,
                                                                  *depth_buffer_,
                                                                  uniform_buffers_,
                                                                  getScene(),
                                                                  is_wire_frame_);

    for (const auto& image_view : swap_chain_->imageViews()) {
        swap_chain_framebuffers_.emplace_back(*image_view, graphics_pipeline_->renderPass());
    }

    command_buffers_ =
        std::make_unique<CommandBuffers>(*command_pool_, static_cast<uint32_t>(swap_chain_framebuffers_.size()));
}

void Application::deleteSwapChain() {
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

void Application::drawFrame() {
    const auto no_timeout = std::numeric_limits<uint64_t>::max();

    auto& in_flight_fence = in_flight_fences_[current_frame_];
    const auto image_available_semaphore = image_available_semaphores_[current_frame_].Handle();
    const auto render_finished_semaphore = render_finished_semaphores_[current_frame_].Handle();

    in_flight_fence.wait(no_timeout);

    uint32_t image_index;
    auto result = vkAcquireNextImageKHR(device_->Handle(),
                                        swap_chain_->Handle(),
                                        no_timeout,
                                        image_available_semaphore,
                                        nullptr,
                                        &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || is_wire_frame_ != graphics_pipeline_->isWireFrame()) {
        recreateSwapChain();
        return;
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        LF_ASSERT(std::runtime_error(std::string("failed to acquire next image (") + toString(result) + ")"));
    }

    const auto command_buffer = command_buffers_->begin(image_index);
    render(command_buffer, image_index);
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

    VkSwapchainKHR swap_chains[] = {swap_chain_->Handle()};
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

    if (result != VK_SUCCESS) {
        LF_ASSERT(std::runtime_error(std::string("failed to present next image (") + toString(result) + ")"));
    }

    current_frame_ = (current_frame_ + 1) % in_flight_fences_.size();
}

void Application::render(VkCommandBuffer command_buffer, uint32_t image_index) {
    std::array<VkClearValue, 2> clear_values = {};
    clear_values[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = graphics_pipeline_->renderPass().Handle();
    render_pass_info.framebuffer = swap_chain_framebuffers_[image_index].Handle();
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = swap_chain_->extent();
    render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
    render_pass_info.pClearValues = clear_values.data();

    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    {
        const auto& scene = getScene();

        VkDescriptorSet descriptor_sets[] = {graphics_pipeline_->descriptorSet(image_index)};
        VkBuffer vertex_buffers[] = {scene.VertexBuffer().Handle()};
        const VkBuffer index_buffer = scene.IndexBuffer().Handle();
        VkDeviceSize offsets[] = {0};

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_->Handle());
        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                graphics_pipeline_->pipelineLayout().Handle(),
                                0,
                                1,
                                descriptor_sets,
                                0,
                                nullptr);
        vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);
        vkCmdBindIndexBuffer(command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);

        uint32_t vertex_offset = 0;
        uint32_t index_offset = 0;

        for (const auto& model : scene.Models()) {
            const auto vertex_count = static_cast<uint32_t>(model.NumberOfVertices());
            const auto index_count = static_cast<uint32_t>(model.NumberOfIndices());

            vkCmdDrawIndexed(command_buffer, index_count, 1, index_offset, vertex_offset, 0);

            vertex_offset += vertex_count;
            index_offset += index_count;
        }
    }
    vkCmdEndRenderPass(command_buffer);
}

void Application::updateUniformBuffer(const uint32_t image_index) {
    uniform_buffers_[image_index].SetValue(getUniformBufferObject(swap_chain_->extent()));
}

void Application::recreateSwapChain() {
    device_->waitIdle();
    deleteSwapChain();
    createSwapChain();
}

}
