#include "pch.h"
#include "Application.h"

#include "Log.h"

#include "ImGui/ImguiLayer.h"

#include "Platform/OpenGL/OpenGLContext.h"
#include "Platform/Optix/OptixErrorCodes.h"
#include "Renderer/Renderer.h"
#include "Renderer/RenderCommand.h"

lift::Application* lift::Application::instance_ = nullptr;

lift::Application::Application() {
	LF_CORE_ASSERT(!instance_, "Application already exists");
	instance_ = this;
	window_ = std::unique_ptr<Window>(Window::Create());
	window_->SetEventCallback(LF_BIND_EVENT_FN(Application::OnEvent));

	InitGraphicsContext();
	InitOptix();

	//window_->SetVSync(false);
	PushOverlay<ImGuiLayer>();
}

lift::Application::~Application() {
	RenderCommand::Shutdown();
}

void lift::Application::Run() {
	SetOptixVariables();
	CreateRenderFrame();
	CreateScene();
	optix_context_->validate();

	while (is_running_) {
		ImGuiLayer::Begin();
		RenderCommand::Clear();

		UpdateOptixVariables();

		// Render
		optix_context_->launch(0, window_->GetWidth(), window_->GetHeight());
		hdr_texture_->Bind();
		pixel_output_buffer_->Bind();
		// Display
		output_shader_->Bind();
		output_shader_->SetTexImage2D(window_->GetWidth(), window_->GetHeight());
		Renderer::Submit(vertex_array_);

		// Update Layers
		for (auto& layer : layer_stack_)
			layer->OnUpdate();

		for (auto& layer : layer_stack_)
			layer->OnImguiRender();
		EndFrame();
	}
}

void lift::Application::InitOptix() {
	optix_context_ = optix::Context::create();

	GetOptixSystemInformation();

	pixel_output_buffer_ = std::make_unique<PixelBuffer>(
		static_cast<float>(window_->GetWidth()) * window_->GetHeight() * sizeof(float) * 4);
	hdr_texture_ = std::make_unique<Texture>();

	ptx_programs_["ray_generation"] = optix_context_->createProgramFromPTXString(
		Util::GetPtxString("res/ptx/ray_generation.ptx"), "ray_generation");
	ptx_programs_["exception"] = optix_context_->createProgramFromPTXString(
		Util::GetPtxString("res/ptx/exception.ptx"), "exception");
	ptx_programs_["miss"] = optix_context_->createProgramFromPTXString(
		Util::GetPtxString("res/ptx/miss.ptx"), "miss_gradient");

	optix_context_->setRayTypeCount(1);
	optix_context_->setEntryPointCount(1);
	optix_context_->setStackSize(1800);
	optix_context_->setMaxTraceDepth(2);
	optix_context_->setPrintEnabled(true);
	optix_context_->setExceptionEnabled(RT_EXCEPTION_ALL, true);

	buffer_output_ = optix_context_->createBufferFromGLBO(RT_BUFFER_OUTPUT, pixel_output_buffer_->id);
	buffer_output_->setFormat(RT_FORMAT_FLOAT4); //RGBA32F
	buffer_output_->setSize(window_->GetWidth(), window_->GetHeight());
}

void lift::Application::InitGraphicsContext() {
	graphics_context_ = std::make_unique<OpenGLContext>(static_cast<GLFWwindow*>(window_->GetNativeWindow()));
	graphics_context_->Init();
	RenderCommand::SetClearColor({1.0f, 0.1f, 1.0f, 0.0f});
}

void lift::Application::SetOptixVariables() {
	optix_context_["sysOutputBuffer"]->set(buffer_output_);
	optix_context_["sysCameraPosition"]->setFloat(0.0f, 0.0f, 0.0f);
	optix_context_["sysCameraU"]->setFloat(1.0f, 0.0f, 0.0f);
	optix_context_["sysCameraV"]->setFloat(0.0f, 1.0f, 0.0f);
	optix_context_["sysCameraW"]->setFloat(0.0f, 0.0f, -1.0f);

	optix_context_["sysColorBottom"]->setFloat(bottom_color_);
	optix_context_["sysColorTop"]->setFloat(top_color_);

	optix_context_->setRayGenerationProgram(0, ptx_programs_["ray_generation"]);
	optix_context_->setExceptionProgram(0, ptx_programs_["exception"]);
	optix_context_->setMissProgram(0, ptx_programs_["miss"]);
}

void lift::Application::UpdateOptixVariables() {
	optix_context_["sysColorBottom"]->setFloat(bottom_color_);
	optix_context_["sysColorTop"]->setFloat(top_color_);
}

void lift::Application::CreateRenderFrame() {

	float quad_vertices [4 * 5] = {
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
	};

	vertex_array_.reset(VertexArray::Create());
	std::shared_ptr<VertexBuffer> vertex_buffer{};
	vertex_buffer.reset(VertexBuffer::Create(quad_vertices, sizeof(quad_vertices)));

	vertex_buffer->SetLayout({
		{ShaderDataType::Float3, "a_Position"},
		{ShaderDataType::Float2, "a_Uv"}
	});

	vertex_array_->AddVertexBuffer(vertex_buffer);
	uint32_t indices[6] = {0, 1, 2, 0, 2, 3};
	std::shared_ptr<IndexBuffer> index_buffer;
	index_buffer.reset(IndexBuffer::Create(indices, sizeof(indices) / sizeof(uint32_t)));
	vertex_array_->SetIndexBuffer(index_buffer);

	output_shader_ = std::make_unique<Shader>("res/shaders/texture_quad");
	output_shader_->Bind();
	output_shader_->SetUniform1i("u_Texture", 0);
}

void lift::Application::CreateScene() {
	optix::Acceleration acceleration_root = optix_context_->createAcceleration(std::string("NoAccel"));
	optix::Group group_root = optix_context_->createGroup();
	group_root->setAcceleration(acceleration_root);
	group_root->setChildCount(0);

	optix_context_["sysTopObject"]->set(group_root);
}

void lift::Application::EndFrame() const {
	ImGuiLayer::End();
	graphics_context_->SwapBuffers();
	window_->OnUpdate();
}

void lift::Application::OnEvent(Event& e) {
	EventDispatcher dispatcher(e);
	dispatcher.Dispatch<WindowCloseEvent>(LF_BIND_EVENT_FN(Application::OnWindowClose));
	dispatcher.Dispatch<WindowResizeEvent>(LF_BIND_EVENT_FN(Application::OnWindowResize));
	dispatcher.Dispatch<WindowMinimizeEvent>(LF_BIND_EVENT_FN(Application::OnWindowMinimize));

	for (auto it = layer_stack_.end(); it != layer_stack_.begin();) {
		(*--it)->OnEvent(e);
		if (e.handled_)
			break;
	}
}

bool lift::Application::OnWindowClose(WindowCloseEvent& e) {
	is_running_ = false;
	LF_CORE_TRACE("Closing Window");
	return true;
}

bool lift::Application::OnWindowResize(WindowResizeEvent& e) {
	if (e.GetHeight() && e.GetWidth()) {
		// Only resize when not minimized
		RenderCommand::Resize(e.GetWidth(), e.GetHeight());
		buffer_output_->setSize(e.GetWidth(), e.GetHeight());
		pixel_output_buffer_->Resize(buffer_output_->getElementSize() * e.GetWidth() * e.GetHeight());
	}
	return true;
}

bool lift::Application::OnWindowMinimize(WindowMinimizeEvent& e) {
	LF_CORE_ERROR("window size: {0} {1}", window_->GetWidth(), window_->GetHeight());
	return true;
}

void lift::Application::GetOptixSystemInformation() {
	unsigned int optix_version;
	OPTIX_CALL(rtGetVersion(&optix_version));

	const auto major = optix_version / 10000;
	const auto minor = (optix_version % 10000) / 100;
	const auto micro = optix_version % 100;
	LF_CORE_INFO("");
	LF_CORE_INFO("Optix Info:");
	LF_CORE_INFO("\tVersion: {0}.{1}.{2}", major, minor, micro);

	const auto number_of_devices = optix_context_.getDeviceCount();
	LF_CORE_INFO("\tNumber of Devices = {0}", number_of_devices);

	for (unsigned int i = 0; i < number_of_devices; ++i) {
		char name[256];
		optix_context_->getDeviceAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name);
		LF_CORE_INFO("\tDevice {0}: {1}", i, name);

		int compute_capability[2] = {0, 0};
		optix_context_->getDeviceAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(
											   compute_capability), &compute_capability);
		LF_CORE_INFO("\t\tCompute Support: {0}.{1}", compute_capability[0], compute_capability[1]);
	}
}
