#pragma once

#include "vulkan/device_procedures.h"
#include "vulkan/ray_tracing_properties.h"
#include <denoiser/denoiser_optix.h>
#include <imgui/imgui_layer.h>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace assets {
class Scene;
class UniformBufferObject;
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

using namespace vulkan;
class Renderer {

  public:
  Renderer(const Instance &instance, bool vsync);
  ~Renderer();

  void beginCommand(assets::Scene &scene, size_t current_frame);
  size_t endCommand(assets::Scene &scene, size_t current_frame, assets::UniformBufferObject &ubo);
  void trace(assets::Scene &scene);
  void render(assets::Scene &scene);
  void render(ImguiLayer &user_interface, const Statistics &statistics);
  void display();

  void traceCommand(VkCommandBuffer command_buffer, uint32_t image_index, assets::Scene &scene);
  void rasterizeCommand(VkCommandBuffer command_buffer, uint32_t image_index, assets::Scene &scene);
  void display(VkCommandBuffer command_buffer, uint32_t image_index);

  void denoiseImage();

  void createRayTracingPipeline(assets::Scene &scene);
  void createSwapChain(assets::Scene &scene);
  void deleteSwapChain();
  void deleteAccelerationStructures();
  void recreateSwapChain(assets::Scene &scene);

  void createAccelerationStructures(assets::Scene &scene);

  void waitDeviceIdle();

  void init(VkPhysicalDevice physical_device, assets::Scene &scene, Instance &instance);
  void setupDenoiser();

  void updateUniformBuffer(uint32_t image_index, assets::UniformBufferObject ubo);

  [[nodiscard]] const class Device &device() const { return *device_; }
  [[nodiscard]] class CommandPool &commandPool() { return *command_pool_; }
  [[nodiscard]] const class SwapChain &swapChain() const { return *swap_chain_; }
  [[nodiscard]] const class DepthBuffer &depthBuffer() const { return *depth_buffer_; }
  [[nodiscard]] const class GraphicsPipeline &graphicsPipeline() const { return *graphics_pipeline_; }
  [[nodiscard]] const class FrameBuffer &swapChainFrameBuffer(const size_t i) const {
    return swap_chain_framebuffers_[i];
  }
  [[nodiscard]] const std::vector<assets::UniformBuffer> &uniformBuffers() const { return uniform_buffers_; }

  void createOutputImage();

  private:
  void createBottomLevelStructures(VkCommandBuffer command_buffer, assets::Scene &scene);
  void createTopLevelStructures(VkCommandBuffer command_buffer, assets::Scene &scene);

  void onDeviceSet();

  private:
  std::unique_ptr<Surface> surface_;
  std::unique_ptr<Device> device_;

  std::unique_ptr<SwapChain> swap_chain_;
  std::unique_ptr<DepthBuffer> depth_buffer_;
  std::unique_ptr<GraphicsPipeline> graphics_pipeline_;
  std::vector<FrameBuffer> swap_chain_framebuffers_;
  std::unique_ptr<CommandPool> command_pool_;
  std::unique_ptr<CommandBuffers> command_buffers_;
  std::vector<Semaphore> image_available_semaphores_;
  std::vector<Semaphore> render_finished_semaphores_;
  std::vector<Fence> in_flight_fences_;

  std::unique_ptr<RayTracingProperties> properties_;
  std::unique_ptr<DeviceProcedures> device_procedures_;
  std::vector<BottomLevelAccelerationStructure> bottom_as_;
  std::unique_ptr<Buffer> bottom_buffer_;
  std::unique_ptr<DeviceMemory> bottom_buffer_memory_;
  std::unique_ptr<Buffer> bottom_scratch_buffer_;
  std::unique_ptr<DeviceMemory> bottom_scratch_buffer_memory_;
  std::vector<TopLevelAccelerationStructure> top_as_;
  std::unique_ptr<Buffer> top_buffer_;
  std::unique_ptr<DeviceMemory> top_buffer_memory_;
  std::unique_ptr<Buffer> top_scratch_buffer_;
  std::unique_ptr<DeviceMemory> top_scratch_buffer_memory_;
  std::unique_ptr<Buffer> instances_buffer_;
  std::unique_ptr<DeviceMemory> instances_buffer_memory_;

  std::unique_ptr<Image> accumulation_image_;
  std::unique_ptr<DeviceMemory> accumulation_image_memory_;
  std::unique_ptr<ImageView> accumulation_image_view_;

  std::unique_ptr<Image> output_image_;
  std::unique_ptr<DeviceMemory> output_image_memory_;
  std::unique_ptr<ImageView> output_image_view_;

  std::unique_ptr<Image> denoised_image_;
  std::unique_ptr<DeviceMemory> denoised_image_memory_;
  std::unique_ptr<ImageView> denoised_image_view_;

  std::unique_ptr<RayTracingPipeline> ray_tracing_pipeline_;
  std::unique_ptr<ShaderBindingTable> shader_binding_table_;

  std::vector<assets::UniformBuffer> uniform_buffers_;

  bool vsync_;
  bool is_wire_frame_;

  VkCommandBuffer current_command_buffer_;
  uint32_t current_image_index_;
  VkSemaphore current_image_available_semaphore_;
  VkSemaphore current_render_finished_semaphore_;

  std::unique_ptr<DenoiserOptix> denoiser_;
};
