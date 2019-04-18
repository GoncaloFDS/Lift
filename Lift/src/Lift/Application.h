#pragma once

#include "Core.h"
#include "Window.h"
#include "LayerStack.h"
#include "Events/ApplicationEvent.h"
#include "ImGui/ImGuiLayer.h"

namespace Lift {

	class LIFT_API Application {
	public:
		Application();
		virtual ~Application() = default;

		void Run();

		void OnEvent(Event& e);

		void PushLayer(Layer* layer);
		void PushOverlay(Layer* overlay);

		inline Window& GetWindow() const;
		static inline Application& Get();

	private:
		bool OnWindowClose(WindowCloseEvent& e);

	private:
		std::unique_ptr<Window> m_window;
		bool m_isRunning = true;
		LayerStack m_layerStack;
		ImGuiLayer* m_imGuiLayer;
		static Application* m_instance;

	};

	// Define by Sandbox
	Application* CreateApplication();
}
