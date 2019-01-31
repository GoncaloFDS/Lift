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

		inline Window& GetWindow();
		static inline Application& Get();

	private:
		bool OnWindowClose(WindowCloseEvent& e);
	
		private:
		std::unique_ptr<Window> _window;
		bool _isRunning = true;
		LayerStack _layerStack;
		static Application* Instance;

	};

	// Define by Sandbox
	Application* CreateApplication();
}



