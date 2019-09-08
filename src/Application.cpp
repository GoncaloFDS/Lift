#include "pch.h"
#include "Application.h"

#include "imgui/ImguiLayer.h"
#include "renderer/RenderCommand.h"
#include "core/os/Input.h"
#include "core/Timer.h"
#include "core/Profiler.h"
#include "platform/windows/WindowsWindow.h"
#include "platform/opengl/OpenGLContext.h"
#include "scene/Scene.h"
#include <optix_stubs.h>
#include "cuda/math_constructors.h"
#include "cuda/vec_math.h"

lift::Application *lift::Application::k_Instance = nullptr;

lift::Application::Application() {
    LF_ASSERT(!k_Instance, "Application already exists");
    k_Instance = this;
    window_ = std::unique_ptr<Window>(Window::create({"Lift Engine", 1280, 720, 0, 28}));
    window_->setEventCallback(LF_BIND_EVENT_FN(Application::onEvent));

    Timer::start();
    initGraphicsContext();
    //renderer_.Init();

    //window_->SetVSync(false);
    pushOverlay<ImGuiLayer>();
}

lift::Application::~Application() {
    RenderCommand::shutdown();
}

void lift::Application::run() {
    Profiler profiler("Application Runtime");
    createScene();
    output_texture_ = std::make_unique<Texture>();

    while (is_running_) {
        Timer::tick();
        ImGuiLayer::begin();
        Input::onUpdate();

        camera_->onUpdate();
        launch_parameters_.camera.eye = makeFloat3(camera_->eye());
        launch_parameters_.camera.u = makeFloat3(camera_->vectorU());
        launch_parameters_.camera.v = makeFloat3(camera_->vectorV());
        launch_parameters_.camera.w = makeFloat3(camera_->vectorW());
        launch_parameters_.handle = scene_.getTraversableHandle();

        // Update Layers
        window_->onUpdate();
        for (auto &layer : layer_stack_)
            layer->onUpdate();

        for (auto &layer : layer_stack_)
            layer->onImguiRender();

        renderer_.launchSubframe(scene_, launch_parameters_, f_size_);
        color_buffer_.download(output_texture_->data());
        output_texture_->setData();

        //End frame
        ImGuiLayer::end();
        graphics_context_->swapBuffers();
    }
}

void lift::Application::initGraphicsContext() {
    graphics_context_ = std::make_unique<OpenGLContext>(static_cast<GLFWwindow *>(window_->getNativeWindow()));
    graphics_context_->init();
    RenderCommand::setClearColor({1.0f, 0.1f, 1.0f, 1.0f});
}

void lift::Application::createScene() {
    Profiler profiler{"Create Scene"};
    scene_.loadFromFile("res/models/DamagedHelmet/glTF/DamagedHelmet.gltf");
    //scene_.LoadFromFile("res/models/Sponza/glTF/Sponza.gltf");
    scene_.finalize();

    OPTIX_CHECK(optixInit())

    camera_ = std::make_unique<Camera>(
        vec3(0.0f, 2.0f, -12.f),
        vec3(0.0f),
        vec3(0.0f, -1.0f, 0.0f),
        36.0f, 1.0f);

    Light::Point l_1{
        {1.0f, 1.0f, 0.8f},
        5.0f,
        {10.0f, 1.0f, 1.0f},
        Light::Falloff::QUADRATIC
    };
    lights_.push_back(l_1);
    Light::Point l_2{
        {0.8f, 0.8f, 0.8f},
        3.0f,
        {-10.0f, -5.0f, 1.0f},
        Light::Falloff::QUADRATIC
    };
    lights_.push_back(l_2);
    Light::Point l_3{
        {0.8f, 0.8f, 0.8f},
        3.0f,
        {0.0f, 40.0f, 0.0f},
        Light::Falloff::NONE
    };
    lights_.push_back(l_3);
    launch_parameters_.lights.count = uint32_t(lights_.size());
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>( &launch_parameters_.lights.data ),
        lights_.size()*sizeof(Light::Point)
    ));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>( launch_parameters_.lights.data ),
        lights_.data(),
        lights_.size()*sizeof(Light::Point),
        cudaMemcpyHostToDevice
    ));

    launch_parameters_.miss_color = makeFloat3(0.1f);
}

void lift::Application::onEvent(Event &e) {
    EventDispatcher dispatcher(e);
    dispatcher.dispatch<WindowCloseEvent>(LF_BIND_EVENT_FN(Application::onWindowClose));
    dispatcher.dispatch<WindowResizeEvent>(LF_BIND_EVENT_FN(Application::onWindowResize));
    dispatcher.dispatch<WindowMinimizeEvent>(LF_BIND_EVENT_FN(Application::onWindowMinimize));

    for (auto it = layer_stack_.end(); it!=layer_stack_.begin();) {
        (*--it)->onEvent(e);
        if (e.handled)
            return;
    }
    dispatcher.dispatch<MouseMovedEvent>(LF_BIND_EVENT_FN(Application::onMouseMove));
    dispatcher.dispatch<MouseScrolledEvent>(LF_BIND_EVENT_FN(Application::onMouseScroll));
    dispatcher.dispatch<KeyPressedEvent>(LF_BIND_EVENT_FN(Application::onKeyPress));
    dispatcher.dispatch<KeyReleasedEvent>(LF_BIND_EVENT_FN(Application::onKeyRelease));

}

bool lift::Application::onWindowClose(WindowCloseEvent &e) {
    is_running_ = false;
    LF_TRACE(e.toString());
    return false;
}

void lift::Application::resize(const ivec2 &size) {
    f_size_ = size;
    color_buffer_.alloc(size.x*size.y);
    accum_buffer_.alloc(size.x*size.y);
    launch_parameters_.frame_buffer = (uchar4 *) color_buffer_.get();
    launch_parameters_.accum_buffer = (float4 *) accum_buffer_.get();
    output_texture_->resize(size);
    camera_->setAspectRatio(float(size.x)/size.y);
}

bool lift::Application::onWindowResize(WindowResizeEvent &e) {
    restartAccumulation();
    if (e.getHeight() && e.getWidth()) {
        // Only resize when not minimized
        RenderCommand::resize(e.getWidth(), e.getHeight());
    }

    return false;
}

bool lift::Application::onWindowMinimize(WindowMinimizeEvent &e) const {
    LF_TRACE(e.toString());
    return false;
}

inline bool lift::Application::onMouseMove(MouseMovedEvent &e) {
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

inline bool lift::Application::onMouseScroll(MouseScrolledEvent &e) {
    camera_->zoom(e.getYOffset()*-10);
    return false;
}

bool lift::Application::onKeyPress(lift::KeyPressedEvent &e) {
    if (e.getKeyCode()==LF_KEY_W) {
        camera_->setMoveDirection(Direction::FORWARD);
    }
    if (e.getKeyCode()==LF_KEY_S) {
        camera_->setMoveDirection(Direction::BACK);
    }
    if (e.getKeyCode()==LF_KEY_D) {
        camera_->setMoveDirection(Direction::RIGHT);
    }
    if (e.getKeyCode()==LF_KEY_A) {
        camera_->setMoveDirection(Direction::LEFT);
    }
    if (e.getKeyCode()==LF_KEY_E) {
        camera_->setMoveDirection(Direction::UP);
    }
    if (e.getKeyCode()==LF_KEY_Q) {
        camera_->setMoveDirection(Direction::DOWN);
    }

    return false;
}
bool lift::Application::onKeyRelease(lift::KeyReleasedEvent &e) {
    if (e.getKeyCode()==LF_KEY_W) {
        camera_->setMoveDirection(Direction::FORWARD, -1.0f);
    }
    if (e.getKeyCode()==LF_KEY_S) {
        camera_->setMoveDirection(Direction::BACK, -1.0f);
    }
    if (e.getKeyCode()==LF_KEY_D) {
        camera_->setMoveDirection(Direction::RIGHT, -1.0f);
    }
    if (e.getKeyCode()==LF_KEY_A) {
        camera_->setMoveDirection(Direction::LEFT, -1.0f);
    }
    if (e.getKeyCode()==LF_KEY_E) {
        camera_->setMoveDirection(Direction::UP, -1.0f);
    }
    if (e.getKeyCode()==LF_KEY_Q) {
        camera_->setMoveDirection(Direction::DOWN, -1.0f);
    }
    return false;
}
