#pragma once

#include <imgui/ui_elements.h>
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

    static Application& get() { return *s_instance; }
    [[nodiscard]] Window& getWindow() const { return *window_; }
    UiElements ui_elements;

private:
    bool is_running_ = true;
    std::unique_ptr<Window> window_;
    std::unique_ptr<GraphicsContext> graphics_context_;
    Renderer renderer_;

    LayerStack layer_stack_;
    std::shared_ptr<Camera> camera_;

    static Application* s_instance;

    void initGraphicsContext();
    void hardcodeSceneEntities(Scene& scene);

    void onEvent(Event& e);
    bool onWindowClose(WindowCloseEvent& e);
    bool onWindowResize(WindowResizeEvent& e);
    bool onWindowMinimize(WindowMinimizeEvent& e) const;
    bool onMouseMove(MouseMovedEvent& e);
    bool onMouseScroll(MouseScrolledEvent& e);
    bool onKeyPress(KeyPressedEvent& e);
    bool onKeyRelease(KeyReleasedEvent& e);

    void onUpdate(const Scene& scene);
    void endFrame();
    void initUiElements();
    void applyUiRequestedChanges();
};

// Defined by Sandbox
std::shared_ptr<Application> createApplication();
}
