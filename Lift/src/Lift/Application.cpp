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

		imgui_layer_ = new ImGuiLayer();
		PushOverlay(imgui_layer_);
	}

	void Application::Run() {

		RTprogram ray_gen_program;
		RTbuffer buffer;

		while (is_running_) {
			glClearColor(1, 0, 1, 1);
			glClear(GL_COLOR_BUFFER_BIT);

			for (Layer* layer : layer_stack_)
				layer->OnUpdate();

			ImGuiLayer::Begin();
			for (Layer* layer : layer_stack_)
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

	void Application::PushLayer(Layer* layer) {
		layer_stack_.PushLayer(layer);
		layer->OnAttach();
	}

	void Application::PushOverlay(Layer* overlay) {
		layer_stack_.PushOverlay(overlay);
		overlay->OnAttach();
	}

	bool Application::OnWindowClose(WindowCloseEvent& e) {
		is_running_ = false;

		return true;
	}
}
