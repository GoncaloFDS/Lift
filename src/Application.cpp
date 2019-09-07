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

lift::Application* lift::Application::instance_ = nullptr;

lift::Application::Application() {
    LF_CORE_ASSERT(!instance_, "Application already exists");
    instance_ = this;
    window_ = std::unique_ptr<Window>(Window::Create({"Lift Engine", 1280, 720, 0, 28}));
    window_->SetEventCallback(LF_BIND_EVENT_FN(Application::OnEvent));

    Timer::Start();
    InitGraphicsContext();
    //renderer_.Init();

    //window_->SetVSync(false);
    PushOverlay<ImGuiLayer>();
}

lift::Application::~Application() {
    RenderCommand::Shutdown();
}

void lift::Application::Run() {
    Profiler profiler("Application Runtime");
    CreateScene();
    output_texture_ = std::make_unique<Texture>();

    while (is_running_) {
        Timer::Tick();
        ImGuiLayer::Begin();
        Input::OnUpdate();

        camera_->OnUpdate();
        launch_parameters_.camera.eye = make_float3(camera_->Eye());
        launch_parameters_.camera.U = make_float3(camera_->VectorU());
        launch_parameters_.camera.V = make_float3(camera_->VectorV());
        launch_parameters_.camera.W = make_float3(camera_->VectorW());
        launch_parameters_.handle = scene_.GetTraversableHandle();

        // Update Layers
        window_->OnUpdate();
        for (auto& layer : layer_stack_)
            layer->OnUpdate();

        for (auto& layer : layer_stack_)
            layer->OnImguiRender();

        renderer_.LaunchSubframe(scene_, launch_parameters_, f_size_);
        color_buffer_.download(output_texture_->Data());
        output_texture_->SetData();

        //End frame
        ImGuiLayer::End();
        graphics_context_->SwapBuffers();
    }
}

void lift::Application::InitGraphicsContext() {
    graphics_context_ = std::make_unique<OpenGLContext>(static_cast<GLFWwindow*>(window_->GetNativeWindow()));
    graphics_context_->Init();
    RenderCommand::SetClearColor({1.0f, 0.1f, 1.0f, 1.0f});
}

void lift::Application::CreateScene() {
    Profiler profiler{"Create Scene"};
    scene_.LoadFromFile("res/models/DamagedHelmet/glTF/DamagedHelmet.gltf");
    //scene_.LoadFromFile("res/models/Sponza/glTF/Sponza.gltf");
    scene_.Finalize();

    OPTIX_CHECK(optixInit())

    camera_ = std::make_unique<Camera>(
        vec3(0.0f, 2.0f, -12.f),
        vec3(0.0f),
        vec3(0.0f, -1.0f, 0.0f),
        36.0f, 1.0f);

    Light::Point l1{
        {1.0f, 1.0f, 0.8f},
        5.0f,
        {10.0f, 1.0f, 1.0f},
        Light::Falloff::QUADRATIC
    };
    lights_.push_back(l1);
    Light::Point l2{
        {0.8f, 0.8f, 0.8f},
        3.0f,
        {-10.0f, -5.0f, 1.0f},
        Light::Falloff::QUADRATIC
    };
    lights_.push_back(l2);
    Light::Point l3{
        {0.8f, 0.8f, 0.8f},
        3.0f,
        {0.0f, 40.0f, 0.0f},
        Light::Falloff::NONE
    };
    lights_.push_back(l3);
    launch_parameters_.lights.count = uint32_t(lights_.size());
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>( &launch_parameters_.lights.data ),
        lights_.size() * sizeof(Light::Point)
    ));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>( launch_parameters_.lights.data ),
        lights_.data(),
        lights_.size() * sizeof(Light::Point),
        cudaMemcpyHostToDevice
    ));

    launch_parameters_.miss_color = make_float3(0.1f);
}

void lift::Application::OnEvent(Event& e) {
    EventDispatcher dispatcher(e);
    dispatcher.Dispatch<WindowCloseEvent>(LF_BIND_EVENT_FN(Application::OnWindowClose));
    dispatcher.Dispatch<WindowResizeEvent>(LF_BIND_EVENT_FN(Application::OnWindowResize));
    dispatcher.Dispatch<WindowMinimizeEvent>(LF_BIND_EVENT_FN(Application::OnWindowMinimize));

    for (auto it = layer_stack_.end(); it != layer_stack_.begin();) {
        (*--it)->OnEvent(e);
        if (e.handled_)
            return;
    }
    dispatcher.Dispatch<MouseMovedEvent>(LF_BIND_EVENT_FN(Application::OnMouseMove));
    dispatcher.Dispatch<MouseScrolledEvent>(LF_BIND_EVENT_FN(Application::OnMouseScroll));
    dispatcher.Dispatch<KeyPressedEvent>(LF_BIND_EVENT_FN(Application::OnKeyPress));
    dispatcher.Dispatch<KeyReleasedEvent>(LF_BIND_EVENT_FN(Application::OnKeyRelease));

}

bool lift::Application::OnWindowClose(WindowCloseEvent& e) {
    is_running_ = false;
    LF_CORE_TRACE(e.ToString());
    return false;
}

void lift::Application::Resize(const ivec2& size) {
    f_size_ = size;
    color_buffer_.alloc(size.x * size.y);
    accum_buffer_.alloc(size.x * size.y);
    launch_parameters_.frame_buffer = (uchar4*) color_buffer_.get();
    launch_parameters_.accum_buffer = (float4*) accum_buffer_.get();
    output_texture_->Resize(size);
    camera_->SetAspectRatio(float(size.x) / size.y);
}

bool lift::Application::OnWindowResize(WindowResizeEvent& e) {
    RestartAccumulation();
    if (e.GetHeight() && e.GetWidth()) {
        // Only resize when not minimized
        RenderCommand::Resize(e.GetWidth(), e.GetHeight());
    }

    return false;
}

bool lift::Application::OnWindowMinimize(WindowMinimizeEvent& e) const {
    LF_CORE_TRACE(e.ToString());
    return false;
}

inline bool lift::Application::OnMouseMove(MouseMovedEvent& e) {
    if (Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_LEFT)) {
        const auto delta = Input::GetMouseDelta();
        camera_->Orbit(-delta.x, -delta.y);
    } else if (Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_MIDDLE)) {
        const auto delta = Input::GetMouseDelta();
        camera_->Strafe(-delta.x, delta.y);
    } else if (Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_RIGHT)) {
        const auto delta = Input::GetMouseDelta();
        camera_->Zoom(delta.y);
    }
    return false;
}

inline bool lift::Application::OnMouseScroll(MouseScrolledEvent& e) {
    camera_->Zoom(e.GetYOffset() * -10);
    return false;
}

bool lift::Application::OnKeyPress(lift::KeyPressedEvent& e) {
    if (e.GetKeyCode() == LF_KEY_W) {
        camera_->SetMoveDirection(Direction::Forward);
    }
    if (e.GetKeyCode() == LF_KEY_S) {
        camera_->SetMoveDirection(Direction::Back);
    }
    if (e.GetKeyCode() == LF_KEY_D) {
        camera_->SetMoveDirection(Direction::Right);
    }
    if (e.GetKeyCode() == LF_KEY_A) {
        camera_->SetMoveDirection(Direction::Left);
    }
    if (e.GetKeyCode() == LF_KEY_E) {
        camera_->SetMoveDirection(Direction::Up);
    }
    if (e.GetKeyCode() == LF_KEY_Q) {
        camera_->SetMoveDirection(Direction::Down);
    }

    return false;
}
bool lift::Application::OnKeyRelease(lift::KeyReleasedEvent& e) {
    if (e.GetKeyCode() == LF_KEY_W) {
        camera_->SetMoveDirection(Direction::Forward, -1.0f);
    }
    if (e.GetKeyCode() == LF_KEY_S) {
        camera_->SetMoveDirection(Direction::Back, -1.0f);
    }
    if (e.GetKeyCode() == LF_KEY_D) {
        camera_->SetMoveDirection(Direction::Right, -1.0f);
    }
    if (e.GetKeyCode() == LF_KEY_A) {
        camera_->SetMoveDirection(Direction::Left, -1.0f);
    }
    if (e.GetKeyCode() == LF_KEY_E) {
        camera_->SetMoveDirection(Direction::Up, -1.0f);
    }
    if (e.GetKeyCode() == LF_KEY_Q) {
        camera_->SetMoveDirection(Direction::Down, -1.0f);
    }
    return false;
}
