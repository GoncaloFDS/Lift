#include "pch.h"
#include "Application.h"

#include "Log.h"

#include <glad/glad.h>
#include "Input.h"

namespace Lift {

#define BIND_EVENT_FN(x) std::bind(&Application::x, this, std::placeholders::_1)

	Application* Application::s_Instance = nullptr;

	Application::Application()	{
		LF_CORE_ASSERT(!s_Instance, "Application already exists");
		s_Instance = this;
		m_window = std::unique_ptr<Window>(Window::Create());
		m_window->SetEventCallback(BIND_EVENT_FN(OnEvent));
	}


	Application::~Application() {
	}

	void Application::Run() {

		while(m_isRunning) {
			glClearColor(1, 0, 1, 1);
			glClear(GL_COLOR_BUFFER_BIT);

			for (Layer* layer : m_layerStack)
				layer->OnUpdate();

			m_window->OnUpdate();	
		}
	}

	void Application::OnEvent(Event& e) {
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<WindowCloseEvent>(BIND_EVENT_FN(OnWindowClose));

		for (auto it = m_layerStack.end(); it != m_layerStack.begin(); ) {
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

	Window& Application::GetWindow() const {
		return *m_window;
	}

	Application& Application::Get() {
		return *s_Instance;
	}

	bool Application::OnWindowClose(WindowCloseEvent& e) {
		m_isRunning = false;
		
		return true;
	}
}
