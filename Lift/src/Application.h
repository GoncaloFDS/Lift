#pragma once

#include "core/os/Window.h"

#include "core/LayerStack.h"
#include "events/ApplicationEvent.h"

#include "renderer/GraphicsContext.h"
#include "scene/cameras/PerspectiveCamera.h"
#include "renderer/Renderer.h"
#include "renderer/Texture.h"

namespace lift {
	class MouseMovedEvent;

	class Application {
	public:
		Application();
		virtual ~Application();

		void Run();

		template <typename T>
		void PushLayer() { layer_stack_.PushLayer<T>(); }

		template <typename T>
		void PushOverlay() { layer_stack_.PushOverlay<T>(); }

		void Resize(const ivec2& size);

		static Application& Get() { return *instance_; }
		[[nodiscard]] Window& GetWindow() const { return *window_; }

		auto GetFrameTextureId() const { return target_texture_->id; }

		void RestartAccumulation() { accumulated_frames_ = 0; }

		vec3 material_albedo_{.3f, .7f, .9f};
	private:
		bool is_running_ = true;
		std::unique_ptr<Window> window_;
		std::unique_ptr<GraphicsContext> graphics_context_;
		Renderer renderer_;

		LayerStack layer_stack_;

		PerspectiveCamera camera_;

		std::unique_ptr<Texture> target_texture_;
		int accumulated_frames_{0};

		//TEMP
		TriangleMesh model_;
		Camera temp_camera_;
		//


		static Application* instance_;

		void InitGraphicsContext();

		void UpdateOptixVariables();

		void CreateScene();
		void CreateLights();
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
