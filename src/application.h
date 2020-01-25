#pragma once

#include "platform/vulkan/frame_buffer.h"
#include "platform/vulkan/window_data.h"
#include "application.h"
#include "platform/vulkan/ray_tracing_properties.h"
#include <vector>
#include <memory>
#include <events/event.h>
#include <events/application_event.h>
#include <events/mouse_event.h>
#include <events/key_event.h>
#include "platform/vulkan/acceleration_structure.h"
#include "scene_list.h"
#include "user_settings.h"

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
    ~Application();

    [[nodiscard]] const std::vector<VkExtensionProperties>& extensions() const;
    [[nodiscard]] const std::vector<VkPhysicalDevice>& physicalDevices() const;

    void setPhysicalDevice(VkPhysicalDevice physical_device);
    void run();

    Application(const UserSettings& user_settings, const WindowData& window_properties, bool vsync);
protected:

    [[nodiscard]] const class Window& window() const { return *window_; }
    [[nodiscard]] const class Device& device() const { return *device_; }
    [[nodiscard]] class CommandPool& commandPool() { return *command_pool_; }
    [[nodiscard]] const class SwapChain& swapChain() const { return *swap_chain_; }
    [[nodiscard]] const class DepthBuffer& depthBuffer() const { return *depth_buffer_; }
    [[nodiscard]] const std::vector<assets::UniformBuffer>& uniformBuffers() const { return uniform_buffers_; }
    [[nodiscard]] const class GraphicsPipeline& graphicsPipeline() const { return *graphics_pipeline_; }
    [[nodiscard]] const class FrameBuffer& swapChainFrameBuffer(const size_t i) const { return swap_chain_framebuffers_[i]; }

    void onDeviceSet();
    void createSwapChain();
    void deleteSwapChain();
    void prepareFrame();
    void drawFrame();
    void render(VkCommandBuffer command_buffer, uint32_t image_index);
    void timeRender(VkCommandBuffer command_buffer, uint32_t image_index);

    void onEvent(Event& event);
    bool onWindowClose(WindowCloseEvent& e);
    bool onWindowResize(WindowResizeEvent& e);
    bool onWindowMinimize(WindowMinimizeEvent& e);
    bool onMouseMove(MouseMovedEvent& e);
    bool onMouseScroll(MouseScrolledEvent& e);
    bool onKeyPress(KeyPressedEvent& e);
    bool onKeyRelease(KeyReleasedEvent& e);

    bool is_wire_frame_{};

    void createAccelerationStructures();
    void deleteAccelerationStructures();
    [[nodiscard]] const assets::Scene& getScene() const { return *scene_; }
    [[nodiscard]] assets::UniformBufferObject getUniformBufferObject(VkExtent2D extent) const;
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
    
    uint32_t scene_index_{};
    UserSettings user_settings_{};
    UserSettings previous_settings_{};
    SceneList::CameraInitialSate camera_initial_sate_{};
    std::unique_ptr<const assets::Scene> scene_;
    std::unique_ptr<class ImguiLayer> user_interface_;
    float camera_x_{};
    float camera_y_{};
    double time_{};
    uint32_t total_number_of_samples_{};
    uint32_t number_of_samples_{};
    bool reset_accumulation_{};

    float mouse_x_{};
    float mouse_y_{};

// Benchmark stats
    double scene_initial_time_{};
    double period_initial_time_{};
    uint32_t period_total_frames_{};
    void loadScene(uint32_t scene_index);
    void checkAndUpdateBenchmarkState(double prev_time);
    void checkFramebufferSize() const;
    void rasterize(VkCommandBuffer command_buffer, uint32_t image_index);
};

}
