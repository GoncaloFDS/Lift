#pragma once

#include "scene_list.h"
#include "user_settings.h"
#include "vulkan/rayTracing/application.h"

class RayTracer final : public vulkan::ray_tracing::Application {
public:
    RayTracer(const UserSettings& user_settings, const vulkan::WindowProperties& window_properties, bool vsync);
    ~RayTracer();

protected:

    [[nodiscard]] const assets::Scene& getScene() const override { return *scene_; }
    [[nodiscard]] assets::UniformBufferObject getUniformBufferObject(VkExtent2D extent) const override;

    void onDeviceSet() override;
    void createSwapChain() override;
    void deleteSwapChain() override;
    void drawFrame() override;
    void render(VkCommandBuffer command_buffer, uint32_t image_index) override;

    void onKey(int key, int scancode, int action, int mods) override;
    void onCursorPosition(double xpos, double ypos) override;
    void onMouseButton(int button, int action, int mods) override;

private:

    void loadScene(uint32_t scene_index);
    void checkAndUpdateBenchmarkState(double prev_time);
    void checkFramebufferSize() const;

    uint32_t scene_index_{};
    UserSettings user_settings_{};
    UserSettings previous_settings_{};
    SceneList::CameraInitialSate camera_initial_sate_{};

    std::unique_ptr<const assets::Scene> scene_;
    std::unique_ptr<class ImguiLayer> user_interface_;

    double mouse_x_{};
    double mouse_y_{};
    bool mouse_left_pressed_{};

    float camera_x_{};
    float camera_y_{};
    double time_{};

    uint32_t total_number_of_samples_{};
    uint32_t number_of_samples_{};
    bool reset_accumulation_{};

    // Benchmark stats
    double scene_initial_time_{};
    double period_initial_time_{};
    uint32_t period_total_frames_{};
};
