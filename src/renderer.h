#pragma once

#include "algorithm_list.h"
#include "vulkan/device_procedures.h"
#include "vulkan/ray_tracing_properties.h"
#include <denoiser/denoiser_optix.h>
#include <imgui/imgui_layer.h>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace assets {
class Scene;
struct UniformBufferObject;
class UniformBuffer;
}  // namespace assets

namespace vulkan {
class CommandBuffers;
class Buffer;
class DeviceMemory;
class Image;
class ImageView;
class Instance;
class Semaphore;
class Fence;
class BottomLevelAccelerationStructure;
class TopLevelAccelerationStructure;
class RayTracingPipeline;
class ShaderBindingTable;
class GraphicsPipeline;
class Surface;
class SwapChain;
class DepthBuffer;
class FrameBuffer;
class CommandPool;
}  // namespace vulkan

class Renderer {

public:
    Renderer(const vulkan::Instance& instance);
    ~Renderer();

    void beginCommand(assets::Scene& scene, size_t current_frame);
    size_t endCommand(assets::Scene& scene, size_t current_frame, assets::UniformBufferObject& ubo);
    void trace(assets::Scene& scene);
    void render(ImguiLayer& user_interface);
    void display();

    void traceCommand(VkCommandBuffer command_buffer, uint32_t image_index, assets::Scene& scene);
    void display(VkCommandBuffer command_buffer, uint32_t image_index);

    void denoiseImage();

    void createSwapChain(assets::Scene& scene, Algorithm algorithm);
    void deleteSwapChain();
    void deleteAccelerationStructures();
    void recreateSwapChain(assets::Scene& scene);

    void createAccelerationStructures(assets::Scene& scene);

    void waitDeviceIdle();

    void init(VkPhysicalDevice physical_device, assets::Scene& scene, vulkan::Instance& instance);
    void setupDenoiser();

    void updateUniformBuffer(uint32_t image_index, assets::UniformBufferObject ubo);

    [[nodiscard]] const vulkan::Device& device() const { return *device_; }
    [[nodiscard]] vulkan::CommandPool& commandPool() { return *command_pool_; }
    [[nodiscard]] const vulkan::SwapChain& swapChain() const { return *swap_chain_; }
    [[nodiscard]] const vulkan::DepthBuffer& depthBuffer() const { return *depth_buffer_; }
    [[nodiscard]] const vulkan::FrameBuffer& swapChainFrameBuffer(const size_t i) const {
        return swap_chain_framebuffers_[i];
    }
    [[nodiscard]] const std::vector<assets::UniformBuffer>& uniformBuffers() const { return uniform_buffers_; }

    void createImages();

    void setDenoised(bool b) { is_denoised = b; }

private:
    void createBottomLevelStructures(VkCommandBuffer command_buffer, assets::Scene& scene);
    void createTopLevelStructures(VkCommandBuffer command_buffer, assets::Scene& scene);
    void createRayTracingPipeline(assets::Scene& scene, Algorithm algorithm);

private:
    std::unique_ptr<vulkan::Surface> surface_;
    std::unique_ptr<vulkan::Device> device_;

    std::unique_ptr<vulkan::SwapChain> swap_chain_;
    std::unique_ptr<vulkan::DepthBuffer> depth_buffer_;
    std::unique_ptr<vulkan::GraphicsPipeline> graphics_pipeline_;
    std::vector<vulkan::FrameBuffer> swap_chain_framebuffers_;
    std::unique_ptr<vulkan::CommandPool> command_pool_;
    std::unique_ptr<vulkan::CommandBuffers> command_buffers_;
    std::vector<vulkan::Semaphore> image_available_semaphores_;
    std::vector<vulkan::Semaphore> render_finished_semaphores_;
    std::vector<vulkan::Fence> in_flight_fences_;

    std::unique_ptr<vulkan::RayTracingProperties> properties_;
    std::unique_ptr<vulkan::DeviceProcedures> device_procedures_;
    std::vector<vulkan::BottomLevelAccelerationStructure> bottom_as_;
    std::unique_ptr<vulkan::Buffer> bottom_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> bottom_buffer_memory_;
    std::unique_ptr<vulkan::Buffer> bottom_scratch_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> bottom_scratch_buffer_memory_;
    std::vector<vulkan::TopLevelAccelerationStructure> top_as_;
    std::unique_ptr<vulkan::Buffer> top_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> top_buffer_memory_;
    std::unique_ptr<vulkan::Buffer> top_scratch_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> top_scratch_buffer_memory_;
    std::unique_ptr<vulkan::Buffer> instances_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> instances_buffer_memory_;

    std::unique_ptr<vulkan::Image> output_image_;
    std::unique_ptr<vulkan::DeviceMemory> output_image_memory_;
    std::unique_ptr<vulkan::ImageView> output_image_view_;

    std::unique_ptr<vulkan::Image> accumulation_image_;
    std::unique_ptr<vulkan::DeviceMemory> accumulation_image_memory_;
    std::unique_ptr<vulkan::ImageView> accumulation_image_view_;

    std::unique_ptr<vulkan::Image> denoised_image_;
    std::unique_ptr<vulkan::DeviceMemory> denoised_image_memory_;
    std::unique_ptr<vulkan::ImageView> denoised_image_view_;

    std::unique_ptr<vulkan::RayTracingPipeline> ray_tracing_pipeline_;
    std::unique_ptr<vulkan::ShaderBindingTable> shader_binding_table_;

    std::vector<assets::UniformBuffer> uniform_buffers_;

    bool vsync_ = false;
    bool is_denoised = false;

    VkCommandBuffer current_command_buffer_{};
    uint32_t current_image_index_{};
    VkSemaphore current_image_available_semaphore_{};
    VkSemaphore current_render_finished_semaphore_{};

    DenoiserOptix denoiser_;
};
