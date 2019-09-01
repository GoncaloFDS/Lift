#pragma once

#include "core/os/Window.h"

#include "core/LayerStack.h"
#include "events/ApplicationEvent.h"

#include "renderer/GraphicsContext.h"
#include "renderer/Renderer.h"
#include "renderer/Texture.h"
#include "scene/cameras/Camera.h"
#include "cuda/launch_parameters.h"
#include "scene/Scene.h"

namespace lift {
class MouseScrolledEvent;
class MouseMovedEvent;

class Application {
public:
	Application();
	virtual ~Application();

	void Run();

	template<typename T>
	void PushLayer() { layer_stack_.PushLayer<T>(); }

	template<typename T>
	void PushOverlay() { layer_stack_.PushOverlay<T>(); }

	void Resize(const ivec2 &size);

	static Application &Get() { return *instance_; }
	[[nodiscard]] Window &GetWindow() const { return *window_; }

	[[nodiscard]] auto GetFrameTextureId() const { return output_texture_->id; }

	void RestartAccumulation() {}

	vec3 material_albedo_{.3f, .7f, .9f};
private:
	bool is_running_ = true;
	std::unique_ptr<Window> window_;
	std::unique_ptr<GraphicsContext> graphics_context_;
	Renderer renderer_;

	LayerStack layer_stack_;

	std::unique_ptr<Camera> camera_;
	Scene scene_;

	std::unique_ptr<Texture> output_texture_;

	//! Temp
	LaunchParameters launch_parameters_;
	CudaBuffer<float4> accum_buffer_;
	CudaBuffer<uchar4> color_buffer_;
	ivec2 f_size_{1000, 1000};
	std::vector<Light::Point> lights_;
	//

	static Application *instance_;

	void InitGraphicsContext();

	void CreateScene();

	void OnEvent(Event &e);
	bool OnWindowClose(WindowCloseEvent &e);
	bool OnWindowResize(WindowResizeEvent &e);
	bool OnWindowMinimize(WindowMinimizeEvent &e) const;
	bool OnMouseMove(MouseMovedEvent &e);
	bool OnMouseScroll(MouseScrolledEvent &e);

};

// Defined by Sandbox
std::shared_ptr<Application> CreateApplication();
}
