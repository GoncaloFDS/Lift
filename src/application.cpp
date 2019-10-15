#include "pch.h"
#include "application.h"

#include "imgui/imgui_layer.h"
#include "renderer/render_command.h"
#include "core/os/input.h"
#include "core/timer.h"
#include "core/profiler.h"
#include "platform/windows/windows_window.h"
#include "platform/opengl/opengl_context.h"
#include "scene/scene.h"
#include <optix_stubs.h>
#include "cuda/math_constructors.h"
#include "cuda/vec_math.h"
#include <thread>
#include <platform/opengl/opengl_display.h>

lift::Application* lift::Application::s_instance = nullptr;

lift::Application::Application() {
    LF_ASSERT(!s_instance, "Application already exists");
    s_instance = this;
    window_ = std::unique_ptr<Window>(Window::create({"Lift Engine", 1280, 720, 0, 28}));
    window_->setEventCallback(LF_BIND_EVENT_FN(Application::onEvent));

    Timer::start();
    initGraphicsContext();

    window_->setVSync(false);
    pushOverlay<ImGuiLayer>();
}

lift::Application::~Application() {
    RenderCommand::shutdown();
}

void lift::Application::run() {
    Scene scene;
    //scene.loadFromFile("res/models/WaterBottle/WaterBottle.gltf");
    scene.loadFromFile("res/models/Sponza/glTF/Sponza.gltf");
    scene.finalize();

    camera_ = scene.camera();
    camera_->setAspectRatio(window_->aspectRatio());
    renderer_.init(CudaOutputBufferType::GL_INTEROP, window_->size());

    hardcodeSceneEntities(scene);

    OpenGLDisplay gl_display;

    while (is_running_) {
        ImGuiLayer::begin();

        onUpdate(scene);
        renderer_.updateLaunchParameters(scene);

        renderer_.launchSubframe(scene, window_->size());

        renderer_.displaySubframe(gl_display, window_->getNativeWindow());

        endFrame();
    }
}

void lift::Application::endFrame() {
    ImGuiLayer::end();
    graphics_context_->swapBuffers();
}

void lift::Application::onUpdate(const lift::Scene& scene) {
    Profiler profiler(Profiler::Id::SceneUpdate);
    Timer::tick();

    // Update Layers
    window_->onUpdate();
    Input::onUpdate();
    for (auto& layer : layer_stack_) {
        layer->onUpdate();
    }

    for (auto& layer : layer_stack_) {
        layer->onImguiRender();
    }

    camera_->onUpdate();
}

void lift::Application::initGraphicsContext() {
    graphics_context_ = std::make_unique<OpenGLContext>(static_cast<GLFWwindow*>(window_->getNativeWindow()));
    graphics_context_->init();
    RenderCommand::setClearColor({1.0f, 0.1f, 1.0f, 1.0f});
}

void lift::Application::onEvent(Event& e) {
    EventDispatcher dispatcher(e);
    dispatcher.dispatch<WindowCloseEvent>(LF_BIND_EVENT_FN(Application::onWindowClose));
    dispatcher.dispatch<WindowResizeEvent>(LF_BIND_EVENT_FN(Application::onWindowResize));
    dispatcher.dispatch<WindowMinimizeEvent>(LF_BIND_EVENT_FN(Application::onWindowMinimize));

    for (auto it = layer_stack_.end(); it != layer_stack_.begin();) {
        (*--it)->onEvent(e);
        if (e.handled)
            return;
    }
    dispatcher.dispatch<MouseMovedEvent>(LF_BIND_EVENT_FN(Application::onMouseMove));
    dispatcher.dispatch<MouseScrolledEvent>(LF_BIND_EVENT_FN(Application::onMouseScroll));
    dispatcher.dispatch<KeyPressedEvent>(LF_BIND_EVENT_FN(Application::onKeyPress));
    dispatcher.dispatch<KeyReleasedEvent>(LF_BIND_EVENT_FN(Application::onKeyRelease));

}

bool lift::Application::onWindowClose(WindowCloseEvent& e) {
    is_running_ = false;
    LF_TRACE(e.toString());
    return false;
}

bool lift::Application::onWindowResize(WindowResizeEvent& e) {
    if (e.height() && e.width()) {
        // Only resize when not minimized
        RenderCommand::resize(e.width(), e.height());
        camera_->setAspectRatio(float(e.width()) / float(e.height()));
        renderer_.onResize(e.width(), e.height());
    }

    return false;
}

bool lift::Application::onWindowMinimize(WindowMinimizeEvent& e) const {
    LF_TRACE(e.toString());
    return false;
}

inline bool lift::Application::onMouseMove(MouseMovedEvent& e) {
    if (Input::isMouseButtonPressed(LF_MOUSE_BUTTON_LEFT)) {
        const auto delta = Input::getMouseDelta();
        camera_->orbit(-delta.x, -delta.y);
    } else if (Input::isMouseButtonPressed(LF_MOUSE_BUTTON_MIDDLE)) {
        const auto delta = Input::getMouseDelta();
        camera_->strafe(-delta.x, delta.y);
    } else if (Input::isMouseButtonPressed(LF_MOUSE_BUTTON_RIGHT)) {
        const auto delta = Input::getMouseDelta();
        camera_->zoom(delta.y);
    }
    return false;
}

inline bool lift::Application::onMouseScroll(MouseScrolledEvent& e) {
    camera_->zoom(e.getYOffset() * -10);
    return false;
}

bool lift::Application::onKeyPress(lift::KeyPressedEvent& e) {
    if (e.getKeyCode() == LF_KEY_W) {
        camera_->setMoveDirection(Direction::FORWARD);
    }
    if (e.getKeyCode() == LF_KEY_S) {
        camera_->setMoveDirection(Direction::BACK);
    }
    if (e.getKeyCode() == LF_KEY_D) {
        camera_->setMoveDirection(Direction::RIGHT);
    }
    if (e.getKeyCode() == LF_KEY_A) {
        camera_->setMoveDirection(Direction::LEFT);
    }
    if (e.getKeyCode() == LF_KEY_E) {
        camera_->setMoveDirection(Direction::UP);
    }
    if (e.getKeyCode() == LF_KEY_Q) {
        camera_->setMoveDirection(Direction::DOWN);
    }

    return false;
}

bool lift::Application::onKeyRelease(lift::KeyReleasedEvent& e) {
    if (e.getKeyCode() == LF_KEY_W) {
        camera_->setMoveDirection(Direction::FORWARD, -1.0f);
    }
    if (e.getKeyCode() == LF_KEY_S) {
        camera_->setMoveDirection(Direction::BACK, -1.0f);
    }
    if (e.getKeyCode() == LF_KEY_D) {
        camera_->setMoveDirection(Direction::RIGHT, -1.0f);
    }
    if (e.getKeyCode() == LF_KEY_A) {
        camera_->setMoveDirection(Direction::LEFT, -1.0f);
    }
    if (e.getKeyCode() == LF_KEY_E) {
        camera_->setMoveDirection(Direction::UP, -1.0f);
    }
    if (e.getKeyCode() == LF_KEY_Q) {
        camera_->setMoveDirection(Direction::DOWN, -1.0f);
    }
    return false;
}

void lift::Application::hardcodeSceneEntities(lift::Scene& scene) {
    const float low_offset = scene.aabb().maxExtent();

    // TODO: add light support to Scene
    Lights::PointLight light0;
    light0.color = {1.0f, 1.0f, 0.8f};
    light0.intensity = 5.0f;
    light0.position = makeFloat3(scene.aabb().center()) + makeFloat3(low_offset);
    light0.falloff = Light::Falloff::QUADRATIC;
    scene.addLight(light0);

    Lights::PointLight light1;
    light1.color = {0.8f, 0.8f, 1.0f};
    light1.intensity = 3.0f;
    light1.position =
        makeFloat3(scene.aabb().center()) + makeFloat3(-low_offset, 0.5f * low_offset, -0.5f * low_offset);
    light1.falloff = Light::Falloff::QUADRATIC;
    scene.addLight(light1);

    renderer_.allocLights(scene);
}
