#include "pch.h"
#include "Application.h"

#include "ImGui/ImguiLayer.h"
#include "Platform/OpenGL/OpenGLContext.h"
#include "Platform/Optix/OptixContext.h"
#include "Renderer/Renderer.h"
#include "Renderer/RenderCommand.h"
#include "Events/MouseEvent.h"
#include "Core/os/Input.h"
#include "Core/Timer.h"
#include "Core/Profiler.h"
#include "Scene/Resources/Mesh.h"
#include "Cuda/material_parameter.cuh"
#include "glad/glad.h"
#include "Scene/Resources/Plane.h"
#include "Scene/Resources/Sphere.h"
#include "Scene/Resources/Parallelogram.h"

lift::Application* lift::Application::instance_ = nullptr;

lift::Application::Application() {
	LF_CORE_ASSERT(!instance_, "Application already exists");
	instance_ = this;
	window_ = std::unique_ptr<Window>(Window::Create({"Lift Engine", 1280, 720, 0, 28}));
	window_->SetEventCallback(LF_BIND_EVENT_FN(Application::OnEvent));

	Timer::Start();
	InitGraphicsContext();
	InitOptix();

	//window_->SetVSync(false);
	PushOverlay<ImGuiLayer>();
}

lift::Application::~Application() {
	RenderCommand::Shutdown();
}

void lift::Application::Run() {
	Profiler profiler("Application Runtime");
	render_frame_.Init(window_->GetWidth(), window_->GetHeight());
	SetOptixVariables();
	//CreateRenderFrame();
	CreateScene();

	while (is_running_) {
		Timer::Tick();
		ImGuiLayer::Begin();
		//RenderCommand::Clear();

		// Update Layers
		for (auto& layer : layer_stack_)
			layer->OnUpdate();

		for (auto& layer : layer_stack_)
			layer->OnImguiRender();


		const auto size = ImGuiLayer::GetRenderWindowSize();
		render_frame_.Resize(static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y));
		camera_.SetViewport(uint32_t(size.x), uint32_t(size.y));

		// Render
		optix_context_["sys_iteration_index"]->setInt(accumulated_frames_);
		render_frame_.Bind();
		UpdateOptixVariables();
		try {
			optix_context_->launch(0, RTsize(size.x), RTsize(size.y));
		}
		catch (optix::Exception& e) {
			LF_ASSERT(false, e.getErrorString());
		}
		accumulated_frames_++;


		//End frame
		ImGuiLayer::End();
		graphics_context_->SwapBuffers();
		window_->OnUpdate();
	}
}

void lift::Application::CreateOptixProgram(const std::string& ptx, const std::string& program_name) {
	ptx_programs_[program_name] = optix_context_->createProgramFromPTXString(
		Util::GetPtxString(ptx.c_str()), program_name);
}

void lift::Application::InitOptix() {
	Profiler profiler("Optix Initialization");
	optix_context_ = OptixContext::Create();

	OptixContext::PrintInfo();

	try {

		CreateOptixProgram("res/ptx/ray_generation.ptx", "ray_generation");
		CreateOptixProgram("res/ptx/exception.ptx", "exception");
		//CreateOptixProgram("res/ptx/miss.ptx", "miss_environment_constant");
		CreateOptixProgram("res/ptx/miss.ptx", "miss_environment_null");
		CreateOptixProgram("res/ptx/triangle_bounding_box.ptx", "triangle_bounding_box");
		CreateOptixProgram("res/ptx/triangle_intersection.ptx", "triangle_intersection");
		CreateOptixProgram("res/ptx/closest_hit.ptx", "closest_hit");
		CreateOptixProgram("res/ptx/closest_hit_light.ptx", "closest_hit_light");
		CreateOptixProgram("res/ptx/any_hit.ptx", "any_hit_shadow");
		CreateOptixProgram("res/ptx/light_sample.ptx", "sample_light_constant");
		CreateOptixProgram("res/ptx/light_sample.ptx", "sample_light_parallelogram");
	}
	catch (optix::Exception& e) {
		LF_CORE_ERROR(e.getErrorString());
	}

	optix_context_->setRayTypeCount(2);
	optix_context_->setEntryPointCount(1);
	optix_context_->setStackSize(1024);
	//optix_context_->setMaxTraceDepth(2);
	optix_context_->setPrintEnabled(true);
	optix_context_->setExceptionEnabled(RT_EXCEPTION_ALL, true);
}

void lift::Application::InitGraphicsContext() {
	graphics_context_ = std::make_unique<OpenGLContext>(static_cast<GLFWwindow*>(window_->GetNativeWindow()));
	graphics_context_->Init();
	RenderCommand::SetClearColor({1.0f, 0.1f, 1.0f, 0.0f});
}

void lift::Application::SetOptixVariables() {
	optix_context_->setRayGenerationProgram(0, ptx_programs_["ray_generation"]);
	optix_context_->setExceptionProgram(0, ptx_programs_["exception"]);
	optix_context_->setMissProgram(0, ptx_programs_["miss"]);

	optix_context_["sys_output_buffer"]->set(render_frame_.GetBufferOutput());
	optix_context_["sys_camera_position"]->setFloat(0.0f, 0.0f, 0.0f);
	optix_context_["sys_camera_u"]->setFloat(1.0f, 0.0f, 0.0f);
	optix_context_["sys_camera_v"]->setFloat(0.0f, 1.0f, 0.0f);
	optix_context_["sys_camera_w"]->setFloat(0.0f, 0.0f, -1.0f);

	optix_context_["sys_scene_epsilon"]->setFloat(500.0f * 1e-7f);
	optix_context_["sys_path_lengths"]->setInt(2, 2);

	optix_context_["sys_iteration_index"]->setInt(0);

}

void lift::Application::UpdateLightParameters() {
	if (!light_definitions_.empty()) {
		void* dst = static_cast<LightDefinition*>(light_definitions_buffer_->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
		memcpy(dst, light_definitions_.data(), sizeof(LightDefinition) * light_definitions_.size());
		light_definitions_buffer_->unmap();
	}

	optix_context_["sys_light_definitions"]->setBuffer(light_definitions_buffer_);
	optix_context_["sys_num_lights"]->setInt(int(light_definitions_.size()));
}

void lift::Application::UpdateOptixVariables() {
	if (camera_.OnUpdate()) {
		optix_context_["sys_camera_position"]->set3fv(value_ptr(camera_.GetPosition()));
		optix_context_["sys_camera_u"]->set3fv(value_ptr(camera_.GetVectorU()));
		optix_context_["sys_camera_v"]->set3fv(value_ptr(camera_.GetVectorV()));
		optix_context_["sys_camera_w"]->set3fv(value_ptr(camera_.GetVectorW()));
	}
	//optix_context_["sys_color_top"]->set3fv(value_ptr(top_color_));
	//optix_context_["sys_color_bottom"]->set3fv(value_ptr(bottom_color_));
	UpdateMaterialParameters(); //TODO check if necessary

	// Update Lights
	UpdateLightParameters();

}

void lift::Application::CreateScene() {
	Profiler profiler{"Create Scene"};
	InitMaterials();
	camera_.SetViewport(window_->GetWidth(), window_->GetHeight());
	camera_.Pan(0.f, 10.f);
	group_root_ = optix_context_->createGroup();
	acceleration_root_ = optix_context_->createAcceleration("Trbvh");
	group_root_->setAcceleration(acceleration_root_);
	group_root_->setChildCount(0);

	optix_context_["sys_top_object"]->set(group_root_);

	Plane plane(1, 1, 1);
	plane.SetMaterial(opaque_material_);
	plane.SetTransform(scale(mat4(1), {5.0f, 5.0f, 5.0f}));
	plane.SubmitMesh(group_root_);

	Sphere sphere(180, 90, 1.0f, M_PIf);
	sphere.SetMaterial(opaque_material_);
	sphere.SetTransform(translate(mat4(1), {0.0f, 1.0f, 0.0f}));
	sphere.SubmitMesh(group_root_);

	Mesh mesh1("res/models/Lantern/glTF/Lantern.gltf");
	mesh1.SetMaterial(opaque_material_);
	mesh1.SetTransform(translate(mat4(1), {-2.0f, 3.0f, 0.0f}));
	mesh1.SubmitMesh(group_root_);

	Mesh mesh2("res/models/Lantern/glTF/Lantern.gltf");
	mesh2.SetMaterial(opaque_material_);
	mesh2.SetTransform(translate(mat4(1), {2.0f, 3.0f, 0.0f}));
	mesh2.SubmitMesh(group_root_);

	CreateLights();
}

void lift::Application::CreateLights() {
	LightDefinition light{
		LIGHT_PARALLELOGRAM,
		optix::make_float3(-0.5f, 7.0f, -0.5f),
		optix::make_float3(1.0f, 0.0f, 0.0f),
		optix::make_float3(0.0f, 0.0f, 1.0f),
		optix::make_float3(0.0f, 0.0f, 1.0f),
		optix::make_float3(1.0f, 1.0f, 1.0f),
		1.0f
	};
	const auto normal = optix::cross(light.vector_u, light.vector_v);
	light.area = optix::length(normal);
	light.normal = normal / light.area;
	light.emission = optix::make_float3(10.0f);

	auto light_index = int(light_definitions_.size());
	light_definitions_.push_back(light);

	Parallelogram light_mesh(light.position, light.vector_u, light.vector_v, light.normal);
	light_mesh.SetMaterial(light_material_);
	light_mesh.SubmitMesh(group_root_);

	light_definitions_buffer_ = optix_context_->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
	light_definitions_buffer_->setElementSize(sizeof(LightDefinition));
	light_definitions_buffer_->setSize(light_definitions_.size());

	UpdateLightParameters();
	light_sample_buffer_ = optix_context_->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 2);
	const auto light_sample = static_cast<int*>(light_sample_buffer_->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
	light_sample[LIGHT_ENVIRONMENT] = ptx_programs_["sample_light_constant"]->getId();
	light_sample[LIGHT_PARALLELOGRAM] = ptx_programs_["sample_light_parallelogram"]->getId();
	light_sample_buffer_->unmap();
	optix_context_["sys_sample_light"]->setBuffer(light_sample_buffer_);

}

void lift::Application::UpdateMaterialParameters() {
	auto dst = static_cast<MaterialParameter*>(material_parameters_buffer_->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
	for (size_t i = 0; i < material_parameters_gui_.size(); i++, dst++) {
		//auto& src = material_parameters_gui_[i];
		dst->albedo.x = material_albedo_.x; //src.albedo;
		dst->albedo.y = material_albedo_.y; //src.albedo;
		dst->albedo.z = material_albedo_.z; //src.albedo;
	}
	material_parameters_buffer_->unmap();
}

void lift::Application::InitMaterials() {

	MaterialParameterGUI parameters;
	parameters.albedo = optix::make_float3(0.2f, 0.3f, 0.0f);
	material_parameters_gui_.push_back(parameters);

	material_parameters_buffer_ = optix_context_->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
	material_parameters_buffer_->setElementSize(sizeof(MaterialParameter));
	material_parameters_buffer_->setSize(material_parameters_gui_.size());

	UpdateMaterialParameters();

	optix_context_["sys_material_parameters"]->setBuffer(material_parameters_buffer_);

	opaque_material_ = optix_context_->createMaterial();
	opaque_material_->setClosestHitProgram(0, ptx_programs_["closest_hit"]);
	opaque_material_->setAnyHitProgram(1, ptx_programs_["any_hit_shadow"]);

	light_material_ = optix_context_->createMaterial();
	light_material_->setClosestHitProgram(0, ptx_programs_["closest_hit_light"]);
	light_material_->setAnyHitProgram(1, ptx_programs_["any_hit_shadow"]);
}

void lift::Application::OnEvent(Event& e) {
	EventDispatcher dispatcher(e);
	dispatcher.Dispatch<WindowCloseEvent>(LF_BIND_EVENT_FN(Application::OnWindowClose));
	dispatcher.Dispatch<WindowResizeEvent>(LF_BIND_EVENT_FN(Application::OnWindowResize));
	dispatcher.Dispatch<WindowMinimizeEvent>(LF_BIND_EVENT_FN(Application::OnWindowMinimize));

	for (auto it = layer_stack_.end(); it != layer_stack_.begin();) {
		(*--it)->OnEvent(e);
		if (e.handled_)
			return;
	}
	dispatcher.Dispatch<MouseMovedEvent>(LF_BIND_EVENT_FN(Application::OnMouseMove));

}

bool lift::Application::OnWindowClose(WindowCloseEvent& e) {
	is_running_ = false;
	LF_CORE_TRACE(e.ToString());
	return false;
}

bool lift::Application::OnWindowResize(WindowResizeEvent& e) {
	RestartAccumulation();
	if (e.GetHeight() && e.GetWidth()) {
		// Only resize when not minimized
		const auto size = ImGuiLayer::GetRenderWindowSize();
		RenderCommand::Resize(e.GetWidth(), e.GetHeight());
		render_frame_.Resize(uint32_t(size.x), uint32_t(size.y));
		camera_.SetViewport(uint32_t(size.x), uint32_t(size.y));
	}

	return false;
}

bool lift::Application::OnWindowMinimize(WindowMinimizeEvent& e) const {
	LF_CORE_TRACE(e.ToString());
	return false;
}

inline bool lift::Application::OnMouseMove(MouseMovedEvent& e) {
	switch (camera_.GetState()) {
	case CameraState::None: {
		if (Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_LEFT)) {
			camera_.SetState(e.GetX(), e.GetY(), CameraState::Orbit);
		}
		else if (Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_RIGHT)) {
			camera_.SetState(e.GetX(), e.GetY(), CameraState::Dolly);
		}
		else if (Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_MIDDLE)) {
			camera_.SetState(e.GetX(), e.GetY(), CameraState::Pan);
		}
		break;
	}
	case CameraState::Orbit: {
		if (!Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_LEFT))
			camera_.SetState(CameraState::None);
		else
			camera_.Orbit(e.GetX(), e.GetY());
		accumulated_frames_ = 0;
		break;
	}
	case CameraState::Dolly: {
		if (!Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_RIGHT))
			camera_.SetState(CameraState::None);
		else
			camera_.Dolly(e.GetX(), e.GetY());
		accumulated_frames_ = 0;
		break;
	}
	case CameraState::Pan: {
		if (!Input::IsMouseButtonPressed(LF_MOUSE_BUTTON_MIDDLE))
			camera_.SetState(CameraState::None);
		else
			camera_.Pan(e.GetX(), e.GetY());
		accumulated_frames_ = 0;
		break;
	}
	default: LF_CORE_ERROR("Invalid Camera State");
	}

	return false;
}
