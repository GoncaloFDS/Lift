#include "pch.h"
#include "Application.h"

#include "Log.h"

#include <glad/glad.h>

//Temporary
#include <optix.h>
#include "optixu/optixpp_namespace.h"

namespace lift {


	Application* Application::instance_ = nullptr;

	Application::Application() {
		LF_CORE_ASSERT(!instance_, "Application already exists");
		instance_ = this;
		window_ = std::unique_ptr<Window>(Window::Create());
		window_->SetEventCallback(LF_BIND_EVENT_FN(Application::OnEvent));

		PushOverlay<ImGuiLayer>();
	}

	void Application::Run() {

		RTprogram ray_gen_program;
		RTbuffer buffer;

		while (is_running_) {
			glClearColor(1, 0, 1, 1);
			glClear(GL_COLOR_BUFFER_BIT);

			for (auto& layer : layer_stack_)
				layer->OnUpdate();

			ImGuiLayer::Begin();
			for (auto& layer : layer_stack_)
				layer->OnImGuiRender();
			ImGuiLayer::End();

			window_->OnUpdate();
		}
	}

	void Application::OnEvent(Event& e) {
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<WindowCloseEvent>(LF_BIND_EVENT_FN(Application::OnWindowClose));

		for (auto it = layer_stack_.end(); it != layer_stack_.begin();) {
			(*--it)->OnEvent(e);
			if (e.handled_)
				break;
		}
	}


	bool Application::OnWindowClose(WindowCloseEvent& e) {
		is_running_ = false;

		return true;
	}
}
