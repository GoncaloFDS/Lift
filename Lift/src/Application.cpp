#include "pch.h"
#include "Application.h"

#include "imgui/ImguiLayer.h"
#include "renderer/RenderCommand.h"
#include "events/MouseEvent.h"
#include "core/os/Input.h"
#include "core/Timer.h"
#include "core/Profiler.h"
#include "platform/windows/WindowsWindow.h"
#include "platform/opengl/OpenGLContext.h"

constexpr auto OPTIX_COMPATIBILITY = 7;
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

lift::Application* lift::Application::instance_ = nullptr;

lift::Application::Application() {
	LF_CORE_ASSERT(!instance_, "Application already exists");
	instance_ = this;
	window_ = std::unique_ptr<Window>(Window::Create({"Lift Engine", 1280, 720, 0, 28}));
	window_->SetEventCallback(LF_BIND_EVENT_FN(Application::OnEvent));

	Timer::Start();
	InitGraphicsContext();
	renderer_.Init();

	//window_->SetVSync(false);
	PushOverlay<ImGuiLayer>();
}

lift::Application::~Application() {
	RenderCommand::Shutdown();
}

void lift::Application::Run() {
	Profiler profiler("Application Runtime");
	CreateScene();
	target_texture_ = std::make_unique<Texture>();

	while (is_running_) {
		Timer::Tick();
		ImGuiLayer::Begin();
		//RenderCommand::Clear();

		// Update Layers
		window_->OnUpdate();
		for (auto& layer : layer_stack_)
			layer->OnUpdate();

		for (auto& layer : layer_stack_)
			layer->OnImguiRender();

		renderer_.Render();
		renderer_.DownloadPixels(target_texture_->Data());
		target_texture_->SetData();

		//camera_.SetViewport(uint32_t(size.x), uint32_t(size.y));

		// Render
		UpdateOptixVariables();

		//End frame
		ImGuiLayer::End();
		graphics_context_->SwapBuffers();
	}
}

void lift::Application::Resize(const ivec2& size) {
	renderer_.Resize(size);
	target_texture_->Resize(size);
}

void lift::Application::InitGraphicsContext() {
	graphics_context_ = std::make_unique<OpenGLContext>(static_cast<GLFWwindow*>(window_->GetNativeWindow()));
	graphics_context_->Init();
	RenderCommand::SetClearColor({1.0f, 0.1f, 1.0f, 1.0f});
}


void lift::Application::UpdateOptixVariables() {
}

void lift::Application::CreateScene() {
	Profiler profiler{"Create Scene"};
}

void lift::Application::CreateLights() {
}

void lift::Application::InitMaterials() {
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

}

bool lift::Application::OnWindowClose(WindowCloseEvent& e) {
	is_running_ = false;
	LF_CORE_TRACE(e.ToString());
	return false;
}

bool lift::Application::OnWindowResize(WindowResizeEvent& e) {
	RestartAccumulation();
	if (e.GetHeight() && e.GetWidth()) {
		// Only resize when not minimized
		const auto size = ImGuiLayer::GetRenderWindowSize();
		RenderCommand::Resize(e.GetWidth(), e.GetHeight());
		camera_.SetViewport(uint32_t(size.x), uint32_t(size.y));
	}

	return false;
}

bool lift::Application::OnWindowMinimize(WindowMinimizeEvent& e) const {
	LF_CORE_TRACE(e.ToString());
	return false;
}

inline bool lift::Application::OnMouseMove(MouseMovedEvent& e) {
	switch (camera_.GetState()) {
	case CameraState::None: {
		if (Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_LEFT)) {
			camera_.SetState(e.GetX(), e.GetY(), CameraState::Orbit);
		}
		else if (Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_RIGHT)) {
			camera_.SetState(e.GetX(), e.GetY(), CameraState::Dolly);
		}
		else if (Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_MIDDLE)) {
			camera_.SetState(e.GetX(), e.GetY(), CameraState::Pan);
		}
		break;
	}
	case CameraState::Orbit: {
		if (!Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_LEFT))
			camera_.SetState(CameraState::None);
		else
			camera_.Orbit(e.GetX(), e.GetY());
		accumulated_frames_ = 0;
		break;
	}
	case CameraState::Dolly: {
		if (!Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_RIGHT))
			camera_.SetState(CameraState::None);
		else
			camera_.Dolly(e.GetX(), e.GetY());
		accumulated_frames_ = 0;
		break;
	}
	case CameraState::Pan: {
		if (!Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_MIDDLE))
			camera_.SetState(CameraState::None);
		else
			camera_.Pan(e.GetX(), e.GetY());
		accumulated_frames_ = 0;
		break;
	}
	default: LF_CORE_ERROR("Invalid Camera State");
	}

	return false;
}
