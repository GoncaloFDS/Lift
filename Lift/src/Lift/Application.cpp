#include "pch.h"
#include "Application.h"

#include "Log.h"

namespace Lift {

#define BIND_EVENT_FN(x) std::bind(&Application::x, this, std::placeholders::_1)

	Application::Application()	{
		_window = std::unique_ptr<Window>(Window::Create());
		_window->SetEventCallback(BIND_EVENT_FN(OnEvent));
	}


	Application::~Application() {
	}

	void Application::Run() {
	
		while(_isRunning) {
			for(Layer* layer : _layerStack) {
				layer->OnUpdate();
			}

			_window->OnUpdate();
		}
	}

	void Application::OnEvent(Event& e) {
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<WindowCloseEvent>(BIND_EVENT_FN(OnWindowClose));

		for (auto it = _layerStack.end(); it != _layerStack.begin(); ) {
			(*--it)->OnEvent(e);
			if(e.Handled)
				break;
		}
	}

	void Application::PushLayer(Layer* layer) {
		_layerStack.PushLayer(layer);
	}

	void Application::PushOverlay(Layer* overlay) {
		_layerStack.PushOverlay(overlay);
	}

	bool Application::OnWindowClose(WindowCloseEvent& e) {
		_isRunning = false;
		
		return true;
	}
}
