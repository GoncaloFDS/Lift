#pragma once

#include "Core/os/Window.h"

#include <optix_world.h>
#include "Core/LayerStack.h"
#include "Events/ApplicationEvent.h"

#include "Renderer/GraphicsContext.h"
#include "Scene/Cameras/PerspectiveCamera.h"

#include "Renderer/RenderFrame.h"
#include "Renderer/FrameBuffer.h"

#include "Cuda/light_definition.cuh"


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
		void CreateOptixProgram(const std::string& ptx, const std::string& program_name);

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
		//optix::Context& GetOptixContext() { return optix_context_; }
		optix::Program& GetOptixProgram(const std::string& name) { return ptx_programs_[name]; }

		void RestartAccumulation() { accumulated_frames_ = 0; }
		unsigned GetRenderedTexture() const { return render_frame_.GetTextureId(); }

		vec3 material_albedo_{.3f, .7f, .9f};
	private:
		bool is_running_ = true;
		std::unique_ptr<Window> window_;
		std::unique_ptr<GraphicsContext> graphics_context_;

		std::map<std::string, optix::Program> ptx_programs_;
		RenderFrame render_frame_;

		optix::Context optix_context_;
		LayerStack layer_stack_;

		PerspectiveCamera camera_;
		int accumulated_frames_{0};

		// Temp
		optix::Group group_root_;
		optix::Material opaque_material_;
		optix::Material light_material_;
		optix::Acceleration acceleration_root_;
		optix::Buffer material_parameters_buffer_;
		optix::Buffer light_definitions_buffer_;
		optix::Buffer light_sample_buffer_;
		std::vector<MaterialParameterGUI> material_parameters_gui_;
		std::vector<LightDefinition> light_definitions_;
		// 

		static Application* instance_;

		void InitOptix();
		void InitGraphicsContext();

		void SetOptixVariables();
		void UpdateLightParameters();
		void UpdateOptixVariables();

		void CreateScene();
		void CreateLights();
		void UpdateMaterialParameters();
		void InitMaterials();

		void OnEvent(Event& e);
		bool OnWindowClose(WindowCloseEvent& e);
		bool OnWindowResize(WindowResizeEvent& e);
		bool OnWindowMinimize(WindowMinimizeEvent& e) const;
		bool OnMouseMove(MouseMovedEvent& e);

	};

	// Defined by Sandbox
	std::shared_ptr<Application> CreateApplication();
}
