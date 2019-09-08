#pragma once

#include "core/os/window.h"

#include "core/layer_stack.h"
#include "events/application_event.h"
#include "events/key_event.h"
#include "events/mouse_event.h"

#include "renderer/graphics_context.h"
#include "renderer/renderer.h"
#include "renderer/texture.h"
#include "scene/cameras/camera.h"
#include "cuda/launch_parameters.h"
#include "scene/scene.h"

namespace lift {

class Application {
public:
    Application();
    virtual ~Application();

    void run();

    template<typename T>
    void pushLayer() { layer_stack_.pushLayer<T>(); }

    template<typename T>
    void pushOverlay() { layer_stack_.pushOverlay<T>(); }

    void resize(const ivec2 &size);

    static Application &get() { return *k_Instance; }
    [[nodiscard]] Window &getWindow() const { return *window_; }

    [[nodiscard]] auto getFrameTextureId() const { return output_texture_->id; }

    void restartAccumulation() {}

    vec3 material_albedo{.3f, .7f, .9f};
private:
    bool is_running_ = true;
    std::unique_ptr<Window> window_;
    std::unique_ptr<GraphicsContext> graphics_context_;
    Renderer renderer_;

    LayerStack layer_stack_;

    std::unique_ptr<Camera> camera_;
    Scene scene_;

    std::unique_ptr<Texture> output_texture_;

    //! Temp
    LaunchParameters launch_parameters_;
    CudaBuffer<float4> accum_buffer_;
    CudaBuffer<uchar4> color_buffer_;
    ivec2 f_size_{1000, 1000};
    std::vector<Light::Point> lights_;
    //

    static Application *k_Instance;

    void initGraphicsContext();

    void createScene();

    void onEvent(Event &e);
    bool onWindowClose(WindowCloseEvent &e);
    bool onWindowResize(WindowResizeEvent &e);
    bool onWindowMinimize(WindowMinimizeEvent &e) const;
    bool onMouseMove(MouseMovedEvent &e);
    bool onMouseScroll(MouseScrolledEvent &e);
    bool onKeyPress(KeyPressedEvent &e);
    bool onKeyRelease(KeyReleasedEvent &e);

};

// Defined by Sandbox
std::shared_ptr<Application> createApplication();
}
