#pragma once

#include "Core.h"
#include "Window.h"
#include "LayerStack.h"
#include "Events/ApplicationEvent.h"

namespace Lift {

	class LIFT_API Application	{
	public:
		Application();
		virtual ~Application();

		void Run();

		void OnEvent(Event& e);

		void PushLayer(Layer* layer);
		void PushOverlay(Layer* overlay);

	private:
		bool OnWindowClose(WindowCloseEvent& e);

		std::unique_ptr<Window> _window;
		bool _isRunning = true;
		LayerStack _layerStack;
	};

	// Define by Sandbox
	Application* CreateApplication();
}



