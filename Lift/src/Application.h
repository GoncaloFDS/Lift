#pragma once

#include "Core.h"
#include "Window.h"
#include "LayerStack.h"
#include "Events/ApplicationEvent.h"
#include "Renderer/Shader.h"
#include "Renderer/Buffer.h"
#include <optix_world.h>
#include "Renderer/VertexArray.h"


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
		std::unique_ptr<Window> window_{};
		std::unique_ptr<GraphicsContext> graphics_context_{};
		optix::Context optix_context_{};
		LayerStack layer_stack_;
		std::map<std::string, optix::Program> ptx_programs_{};

		// Temp
		std::shared_ptr<VertexArray> vertex_array_{};
		std::shared_ptr<Shader> shader_{};
		// 

		optix::GeometryInstance CreateParallelogram(const optix::float3& anchor, const optix::float3& offset1,
													const optix::float3& offset2);
		void UpdateCamera();
		static Application* instance_;

		void InitOptix();
		void InitGraphicsContext();
		void InitPrograms();

		void CreateScene();
		void EndFrame() const;
		void OnEvent(Event& e);
		bool OnWindowClose(WindowCloseEvent& e);

		void GetOptixSystemInformation();
	};


	// Defined by Sandbox
	std::shared_ptr<Application> CreateApplication();
}
