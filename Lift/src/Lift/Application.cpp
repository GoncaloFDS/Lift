#include "pch.h"
#include "Application.h"

#include "Log.h"

#include <glad/glad.h>

//Temporary
#include <optix.h>
#include "optixu/optixpp_namespace.h"

namespace Lift {

	#define BIND_EVENT_FN(x) std::bind(&Application::x, this, std::placeholders::_1)

	Application* Application::m_instance = nullptr;

	Application::Application() {
		LF_CORE_ASSERT(!m_instance, "Application already exists");
		m_instance = this;
		m_window = std::unique_ptr<Window>(Window::Create());
		m_window->SetEventCallback(BIND_EVENT_FN(OnEvent));

		m_imGuiLayer = new ImGuiLayer();
		PushOverlay(m_imGuiLayer);
	}

	void Application::Run() {

		/*Primary RTAPI objects*/
		RTprogram rayGenProgram;
		RTbuffer buffer;

		while(m_isRunning) {
			glClearColor(1, 0, 1, 1);
			glClear(GL_COLOR_BUFFER_BIT);

			for(Layer* layer : m_layerStack)
				layer->OnUpdate();

			m_imGuiLayer->Begin();
			for(Layer* layer : m_layerStack)
				layer->OnImGuiRender();
			m_imGuiLayer->End();

			m_window->OnUpdate();
		}
	}

	void Application::OnEvent(Event& e) {
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<WindowCloseEvent>(BIND_EVENT_FN(OnWindowClose));

		for(auto it = m_layerStack.end(); it != m_layerStack.begin();) {
			(*--it)->OnEvent(e);
			if(e.Handled)
				break;
		}
	}

	void Application::PushLayer(Layer* layer) {
		m_layerStack.PushLayer(layer);
		layer->OnAttach();
	}

	void Application::PushOverlay(Layer* overlay) {
		m_layerStack.PushOverlay(overlay);
		overlay->OnAttach();

	}

	bool Application::OnWindowClose(WindowCloseEvent& e) {
		m_isRunning = false;

		return true;
	}
}
