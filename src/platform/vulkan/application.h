#pragma once

#include "frame_buffer.h"
#include "window_properties.h"
#include "platform/vulkan/application.h"
#include "ray_tracing_properties.h"
#include <vector>
#include <memory>
#include <events/event.h>
#include <events/application_event.h>
#include <events/mouse_event.h>
#include <events/key_event.h>
#include "acceleration_structure.h"

using namespace lift;

namespace assets {
class Scene;
class UniformBufferObject;
class UniformBuffer;
}

namespace vulkan {
class CommandBuffers;
class Buffer;
class DeviceMemory;
class Image;
class ImageView;
}

namespace vulkan {


class Application {

public:
    virtual ~Application();

    [[nodiscard]] const std::vector<VkExtensionProperties>& extensions() const;
    [[nodiscard]] const std::vector<VkPhysicalDevice>& physicalDevices() const;

    void setPhysicalDevice(VkPhysicalDevice physical_device);
    void run();

    Application(const WindowProperties& window_properties, const bool vsync);
protected:

    [[nodiscard]] const class Window& window() const { return *window_; }
    [[nodiscard]] const class Device& device() const { return *device_; }
    [[nodiscard]] class CommandPool& commandPool() { return *command_pool_; }
    [[nodiscard]] const class SwapChain& swapChain() const { return *swap_chain_; }
    [[nodiscard]] const class DepthBuffer& depthBuffer() const { return *depth_buffer_; }
    [[nodiscard]] const std::vector<assets::UniformBuffer>& uniformBuffers() const { return uniform_buffers_; }
    [[nodiscard]] const class GraphicsPipeline& graphicsPipeline() const { return *graphics_pipeline_; }
    [[nodiscard]] const class FrameBuffer& swapChainFrameBuffer(const size_t i) const { return swap_chain_framebuffers_[i]; }

    [[nodiscard]] virtual const assets::Scene& getScene() const = 0;
    [[nodiscard]] virtual assets::UniformBufferObject getUniformBufferObject(VkExtent2D extent) const = 0;

    virtual void onDeviceSet();
    virtual void createSwapChain();
    virtual void deleteSwapChain();
    virtual void drawFrame();
    virtual void render(VkCommandBuffer command_buffer, uint32_t image_index);

    virtual void onEvent(Event& event);
    virtual bool onWindowClose(WindowCloseEvent& e);
    virtual bool onWindowResize(WindowResizeEvent& e);
    virtual bool onWindowMinimize(WindowMinimizeEvent& e);
    virtual bool onMouseMove(MouseMovedEvent& e);
    virtual bool onMouseScroll(MouseScrolledEvent& e);
    virtual bool onKeyPress(KeyPressedEvent& e);
    virtual bool onKeyRelease(KeyReleasedEvent& e);

    virtual void onKey(int key, int scancode, int action, int mods) {}
    virtual void onCursorPosition(double xpos, double ypos) {}
    virtual void onMouseButton(int button, int action, int mods) {}

    bool is_wire_frame_{};

    void createAccelerationStructures();
    void deleteAccelerationStructures();
private:

    void updateUniformBuffer(uint32_t image_index);
    void recreateSwapChain();
    void createBottomLevelStructures(VkCommandBuffer command_buffer);
    void createTopLevelStructures(VkCommandBuffer command_buffer);
    void createOutputImage();

    const bool vsync_;
    std::unique_ptr<class Window> window_;
    std::unique_ptr<class Instance> instance_;
    std::unique_ptr<class Surface> surface_;
    std::unique_ptr<class Device> device_;
    std::unique_ptr<class SwapChain> swap_chain_;
    std::vector<assets::UniformBuffer> uniform_buffers_;
    std::unique_ptr<class DepthBuffer> depth_buffer_;
    std::unique_ptr<class GraphicsPipeline> graphics_pipeline_;
    std::vector<class FrameBuffer> swap_chain_framebuffers_;
    std::unique_ptr<class CommandPool> command_pool_;
    std::unique_ptr<class CommandBuffers> command_buffers_;
    std::vector<class Semaphore> image_available_semaphores_;
    std::vector<class Semaphore> render_finished_semaphores_;
    std::vector<class Fence> in_flight_fences_;

    std::unique_ptr<class RayTracingProperties> properties_;
    std::unique_ptr<class DeviceProcedures> device_procedures_;
    std::vector<class BottomLevelAccelerationStructure> bottom_as_;
    std::unique_ptr<Buffer> bottom_buffer_;
    std::unique_ptr<DeviceMemory> bottom_buffer_memory_;
    std::unique_ptr<Buffer> bottom_scratch_buffer_;
    std::unique_ptr<DeviceMemory> bottom_scratch_buffer_memory_;
    std::vector<class TopLevelAccelerationStructure> top_as_;
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
    std::unique_ptr<class RayTracingPipeline> ray_tracing_pipeline_;
    std::unique_ptr<class ShaderBindingTable> shader_binding_table_;

    size_t current_frame_{};
    bool is_running_{};
};

}
