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

    static auto get() -> Application& { return *s_instance; }
    [[nodiscard]] auto getWindow() const -> Window& { return *window_; }
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
    auto onWindowClose(WindowCloseEvent& e) -> bool;
    auto onWindowResize(WindowResizeEvent& e) -> bool;
    auto onWindowMinimize(WindowMinimizeEvent& e) -> bool ;
    auto onMouseMove(MouseMovedEvent& e) -> bool;
    auto onMouseScroll(MouseScrolledEvent& e) -> bool;
    auto onKeyPress(KeyPressedEvent& e) -> bool;
    auto onKeyRelease(KeyReleasedEvent& e) -> bool;

    void onUpdate(const Scene& scene);
    void endFrame();
    void initUiElements();
    void applyUiRequestedChanges();
};

// Defined by Sandbox
auto createApplication() -> std::shared_ptr<Application>;
}
