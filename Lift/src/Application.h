#pragma once

#include "Core/os/Window.h"

#include <optix_world.h>
#include "Core/LayerStack.h"
#include "Events/ApplicationEvent.h"

#include "Renderer/Shader.h"
#include "Renderer/VertexArray.h"
#include "Renderer/Texture.h"
#include "Renderer/PerspectiveCamera.h"
#include "Renderer/GraphicsContext.h"

#include "Platform/OpenGL/PixelBuffer.h"


namespace lift {
	class MouseMovedEvent;

	struct MaterialParameterGUI {
		optix::float3 albedo = optix::make_float3(0.5f);
	};

	class Application {
	public:
		Application();
		virtual ~Application();

		void Run();

		template <typename T>
		void PushLayer() {
			layer_stack_.PushLayer<T>();
		}

		template <typename T>
		void PushOverlay() {
			layer_stack_.PushOverlay<T>();
		}

		static Application& Get() { return *instance_; }
		[[nodiscard]] Window& GetWindow() const { return *window_; }
		optix::Context& GetOptixContext() { return optix_context_; }
		optix::Program& GetOptixProgram(const std::string& name) { return ptx_programs_[name]; }

		vec3& GetTopColor() { return top_color_; }
		vec3& GetBottomColor() { return bottom_color_; }

	private:
		bool is_running_ = true;
		std::unique_ptr<Window> window_;
		std::unique_ptr<GraphicsContext> graphics_context_;
		std::unique_ptr<PixelBuffer> pixel_output_buffer_;
		std::unique_ptr<Texture> hdr_texture_;

		std::map<std::string, optix::Program> ptx_programs_;

		optix::Context optix_context_;
		LayerStack layer_stack_;

		PerspectiveCamera camera_;
		int accumulated_frames_{0};

		// Temp
		std::shared_ptr<VertexArray> vertex_array_;
		std::shared_ptr<Shader> output_shader_;
		optix::Buffer buffer_output_;
		vec3 top_color_{1.f, 0.f, 0.f};
		vec3 bottom_color_{1.f, 0.f, 1.f};
		optix::Material opaque_material_;
		optix::Acceleration acceleration_root_;
		optix::Buffer buffer_material_parameters_;
		std::vector<MaterialParameterGUI> gui_material_parameters_;
		// 

		static Application* instance_;

		void InitOptix();
		void InitGraphicsContext();

		void SetOptixVariables();
		void UpdateOptixVariables();

		void CreateRenderFrame();
		void CreateScene();
		void UpdateMaterialParameters();
		void InitMaterials();
		void EndFrame() const;
		void OnEvent(Event& e);
		bool OnWindowClose(WindowCloseEvent& e);
		bool OnWindowResize(WindowResizeEvent& e);
		bool OnWindowMinimize(WindowMinimizeEvent& e) const;
		bool OnMouseMove(MouseMovedEvent& e);

		void GetOptixSystemInformation();

	};

	// Defined by Sandbox
	std::shared_ptr<Application> CreateApplication();
}
