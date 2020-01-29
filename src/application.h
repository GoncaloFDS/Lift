#pragma once

#include <vector>
#include <memory>
#include <assets/camera.h>
#include <imgui/imgui_layer.h>
#include "vulkan/frame_buffer.h"
#include "vulkan/window_data.h"
#include "vulkan/ray_tracing_properties.h"
#include "events/event.h"
#include "events/application_event.h"
#include "events/mouse_event.h"
#include "events/key_event.h"
#include "vulkan/acceleration_structure.h"
#include "scene_list.h"
#include "user_settings.h"

using namespace vulkan;

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
}

namespace lift {

class Application {
public:
    Application(const UserSettings& user_settings, const WindowData& window_properties, bool vsync);
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
    void checkAndUpdateBenchmarkState(double prev_time);

    void onUpdate();
    void deleteSwapChain();
    void createSwapChain();

private:
    const bool vsync_;
    std::unique_ptr<Window> window_;
    std::unique_ptr<Instance> instance_;
    std::unique_ptr<class Renderer> renderer_;
    std::unique_ptr<Camera> camera_;

    size_t current_frame_{};
    bool is_running_{};
    bool is_wire_frame_{};

    uint32_t scene_index_{};
    UserSettings user_settings_{};
    UserSettings previous_settings_{};
    SceneList::CameraInitialSate camera_initial_sate_{};
    std::unique_ptr<assets::Scene> scene_;
    std::unique_ptr<ImguiLayer> user_interface_;
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

};

}
