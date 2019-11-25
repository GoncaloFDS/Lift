#include "pch.h"
#include "application.h"

#include "imgui/imgui_layer.h"
#include "core/os/input.h"
#include "core/timer.h"
#include "core/profiler.h"
#include "cuda/vec_math.h"

lift::Application* lift::Application::s_instance = nullptr;

lift::Application::Application() {
    LF_ASSERT(!s_instance, "Application already exists");
    s_instance = this;
    window_ = std::shared_ptr<Window>(Window::create(
        WindowProperties{"Lift Engine", 1280, 720, 0, 28}));
    window_->setEventCallback(LF_BIND_EVENT_FN(Application::onEvent));

    Timer::start();
    initGraphicsContext();

    window_->setVSync(false);
    pushOverlay<ImGuiLayer>();
}

lift::Application::~Application() {
}

void lift::Application::run() {
    Scene scene;

    scene.loadFromFile("res/models/Sponza/glTF/Sponza.gltf");

    renderer_.init(CudaOutputBufferType::GL_INTEROP, window_);
	renderer_.initOptixContext(scene);

	scene.calculateAabb();

	camera_ = scene.camera();
	camera_->setAspectRatio(window_->aspectRatio());

	hardcodeSceneEntities(scene);

    initUiElements();
    while (is_running_) {
        ImGuiLayer::begin();

        onUpdate(scene);
        renderer_.updateLaunchParameters(scene);

        renderer_.launchSubframe(scene, window_->size());

        renderer_.displaySubframe(window_->getNativeWindow());

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

    if (camera_->onUpdate()) {
        renderer_.resetFrame();
    }
    if (ui_elements.dirty) {
        applyUiRequestedChanges();
    }
}

void lift::Application::applyUiRequestedChanges() {
    renderer_.setClearColor(this->ui_elements.clear_color);
    ui_elements.dirty = false;
}

void lift::Application::initGraphicsContext() {
    graphics_context_ = std::make_unique<OpenGLContext>(window_);
    graphics_context_->init();
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

auto lift::Application::onWindowClose(WindowCloseEvent& e) -> bool {
    is_running_ = false;
    LF_TRACE(e.toString());
    return false;
}

auto lift::Application::onWindowResize(WindowResizeEvent& e) -> bool {
    if (e.height() && e.width()) {
        // Only resize when not minimized
        //RenderCommand::resize(e.width(), e.height());
        camera_->setAspectRatio(float(e.width()) / float(e.height()));
        renderer_.onResize(e.width(), e.height());
    }

    return false;
}

auto lift::Application::onWindowMinimize(WindowMinimizeEvent& e) -> bool {
    LF_TRACE(e.toString());
    return false;
}

inline auto lift::Application::onMouseMove(MouseMovedEvent& e) -> bool {
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

inline auto lift::Application::onMouseScroll(MouseScrolledEvent& e) -> bool {
    camera_->zoom(e.getYOffset() * -10);
    return false;
}

auto lift::Application::onKeyPress(lift::KeyPressedEvent& e) -> bool {
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

auto lift::Application::onKeyRelease(lift::KeyReleasedEvent& e) -> bool {
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
    Light light1;
	light1.position = {-1200.0f, 800, -200};
	light1.v1 = {400.0f, 0.0f, 0.0f};
	light1.v2 = {0.0f, 0.0f, 400.0f};
	light1.normal = normalize(cross(light1.v1, light1.v2));
	light1.emission = {15.0f, 15.0f, 15.0f};
    scene.addLight(light1);

    renderer_.allocLights(scene);
}

void lift::Application::initUiElements() {
    ui_elements.clear_color = renderer_.clearColor();
}
