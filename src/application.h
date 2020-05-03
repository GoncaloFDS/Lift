#pragma once

#include "events/application_event.h"
#include "events/event.h"
#include "events/key_event.h"
#include "events/mouse_event.h"
#include "scene_list.h"
#include "user_settings.h"
#include "vulkan/acceleration_structure.h"
#include "vulkan/frame_buffer.h"
#include "vulkan/ray_tracing_properties.h"
#include "vulkan/window_data.h"
#include <assets/Lights.h>
#include <assets/camera.h>
#include <denoiser/denoiser_optix.h>
#include <imgui/imgui_layer.h>
#include <memory>
#include <vector>

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
class Window;
}  // namespace vulkan

class Application {
public:
    Application(const UserSettings& user_settings, const vulkan::WindowData& window_properties, bool vsync);
    ~Application();

    [[nodiscard]] const std::vector<VkExtensionProperties>& extensions() const;
    [[nodiscard]] const std::vector<VkPhysicalDevice>& physicalDevices() const;

    [[nodiscard]] assets::UniformBufferObject getUniformBufferObject(VkExtent2D extent) const;

    void setPhysicalDevice(VkPhysicalDevice physical_device);
    void run();

private:
    void onEvent(Event& event);
    bool onWindowClose(WindowCloseEvent& e);
    bool onWindowResize(WindowResizeEvent& e);
    bool onWindowMinimize(WindowMinimizeEvent& e);
    bool onMouseMove(MouseMovedEvent& e);
    bool onMouseScroll(MouseScrolledEvent& e);
    bool onKeyPress(KeyPressedEvent& e);
    bool onKeyRelease(KeyReleasedEvent& e);

    void loadScene(uint32_t scene_index);

    void onUpdate();
    void deleteSwapChain();
    void createSwapChain();

    const bool vsync_;
    std::unique_ptr<vulkan::Window> window_;
    std::unique_ptr<class Renderer> renderer_;

    std::unique_ptr<vulkan::Instance> instance_;
    std::unique_ptr<Camera> camera_;
    CameraState camera_initial_state_ {};

    size_t current_frame_ {};
    bool is_running_ {};

    uint32_t scene_index_ {};
    UserSettings user_settings_ {};
    UserSettings previous_settings_ {};
    std::unique_ptr<assets::Scene> scene_;
    std::unique_ptr<ImguiLayer> user_interface_;
    double time_ {};
    uint32_t total_number_of_samples_ {};
    uint32_t number_of_samples_ {};
    uint32_t number_of_frames_ {};
    bool reset_accumulation_ {};

    ParallelogramLight light_;

};
