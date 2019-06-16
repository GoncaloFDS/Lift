#pragma once

#include "Core.h"
#include "Window.h"
#include "LayerStack.h"
#include "Events/ApplicationEvent.h"
#include "Renderer/Shader.h"
#include "Renderer/Buffer.h"
#include <optix_world.h>


namespace lift {

	class Application {
	public:
		Application();
		virtual ~Application() = default;

		void Run();

		template <typename T>
		void PushLayer() {
			layer_stack_.PushLayer<T>();
		}

		template <typename T>
		void PushOverlay() {
			layer_stack_.PushOverlay<T>();
		}

		Window& GetWindow() const { return *window_; }
		static Application& Get() { return *instance_; }

	private:
		bool is_running_ = true;
		std::unique_ptr<Window> window_ {};
		std::unique_ptr<GraphicsContext> graphics_context_ {};
		std::unique_ptr<optix::Context> optix_context_ {};
		LayerStack layer_stack_;

		// Temp
		unsigned int vertex_array_ {};
		std::unique_ptr<VertexBuffer> vertex_buffer_ {};
		std::unique_ptr<IndexBuffer> index_buffer_ {};
		std::unique_ptr<Shader> shader_ {};

		static Application* instance_;

		void InitOptix();
		void InitGraphicsContext();

		void CreateScene();
		void Render();
		void Display();
		void StartFrame();
		void EndFrame();
		void OnEvent(Event& e);
		bool OnWindowClose(WindowCloseEvent& e);

	};


	// Defined by Sandbox
	std::shared_ptr<Application> CreateApplication();
}
