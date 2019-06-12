#pragma once

#include "Core.h"
#include "Window.h"
#include "LayerStack.h"
#include "Events/ApplicationEvent.h"
#include "ImGui/ImGuiLayer.h"

namespace lift {

	class Application {
	public:
		Application();
		virtual ~Application() = default;

		void Run();

		void OnEvent(Event& e);

		void PushLayer(Layer* layer);
		void PushOverlay(Layer* overlay);

		Window& GetWindow() const { return *window_; }
		static Application& Get() { return *instance_; }

	private:
		bool OnWindowClose(WindowCloseEvent& e);

	private:
		std::unique_ptr<Window> window_;
		bool is_running_ = true;
		LayerStack layer_stack_;
		ImGuiLayer* imgui_layer_;
		static Application* instance_;

	};

	// Define by Sandbox
	Application* CreateApplication();
}
