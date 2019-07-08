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

void lift::Application::Run() {
	CreateScene();
	optix_context_->validate();

	while (is_running_) {
		ImGuiLayer::Begin();
		RenderCommand::Clear();

		// Render
		optix_context_->launch(0, window_->GetWidth(), window_->GetHeight());
		hdr_texture_->Bind();
		pbo_output_buffer_->Bind();
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

void lift::Application::CreateScene() {

	float vertices [4 * 5] = {
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
	};

	vertex_array_.reset(VertexArray::Create());
	std::shared_ptr<VertexBuffer> vertex_buffer{};
	vertex_buffer.reset(VertexBuffer::Create(vertices, sizeof(vertices)));

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

void lift::Application::InitOptix() {
	GetOptixSystemInformation();

	pbo_output_buffer_ = std::make_unique<PixelBuffer>(
		static_cast<float>(window_->GetWidth()) * window_->GetHeight() * sizeof(float) * 4);
	hdr_texture_ = std::make_unique<Texture>();

	optix_context_ = optix::Context::create();

	InitPrograms();

	optix_context_->setRayTypeCount(0);
	optix_context_->setEntryPointCount(1);
	optix_context_->setStackSize(1800);
	optix_context_->setMaxTraceDepth(2);
	optix_context_->setPrintEnabled(true);
	optix_context_->setExceptionEnabled(RT_EXCEPTION_ALL, true);

	optix_context_["sysColorBackground"]->setFloat(0.46f, 0.72f, 0.0f);

	buffer_output_ = optix_context_->createBufferFromGLBO(RT_BUFFER_OUTPUT, pbo_output_buffer_->id);
	buffer_output_->setFormat(RT_FORMAT_FLOAT4); //RGBA32F
	buffer_output_->setSize(window_->GetWidth(), window_->GetHeight());
	optix_context_["sysOutputBuffer"]->set(buffer_output_);

	optix_context_->setRayGenerationProgram(0, ptx_programs_["raygeneration"]);
	optix_context_->setExceptionProgram(0, ptx_programs_["exception"]);
}

void lift::Application::InitPrograms() {
	std::string ptx_string = Util::GetPtxString("res/ptx/raygeneration.ptx");
	ptx_programs_["raygeneration"] = optix_context_->createProgramFromPTXString(ptx_string, "raygeneration");

	ptx_string = Util::GetPtxString("res/ptx/exception.ptx");
	ptx_programs_["exception"] = optix_context_->createProgramFromPTXString(ptx_string, "exception");
}

void lift::Application::InitGraphicsContext() {
	graphics_context_ = std::make_unique<OpenGLContext>(static_cast<GLFWwindow*>(window_->GetNativeWindow()));
	graphics_context_->Init();

	RenderCommand::SetClearColor({1.0f, 0.1f, 1.0f, 0.0f});
}

void lift::Application::EndFrame() const {
	ImGuiLayer::End();
	graphics_context_->SwapBuffers();
	window_->OnUpdate();
}

void lift::Application::OnEvent(Event& e) {
	EventDispatcher dispatcher(e);
	dispatcher.Dispatch<WindowCloseEvent>(LF_BIND_EVENT_FN(Application::OnWindowClose));

	for (auto it = layer_stack_.end(); it != layer_stack_.begin();) {
		(*--it)->OnEvent(e);
		if (e.handled_)
			break;
	}
}

bool lift::Application::OnWindowClose(WindowCloseEvent& e) {
	is_running_ = false;

	return true;
}

void lift::Application::GetOptixSystemInformation() {
	unsigned int optix_version;
	OPTIX_CALL(rtGetVersion(&optix_version));

	const unsigned int major = optix_version / 10000;
	const unsigned int minor = (optix_version % 10000) / 100;
	const unsigned int micro = optix_version % 100;
	LF_CORE_INFO("");
	LF_CORE_INFO("Optix Info:");
	LF_CORE_INFO("\tVersion: {0}.{1}.{2}", major, minor, micro);

	unsigned int number_of_devices = 0;
	OPTIX_CALL(rtDeviceGetDeviceCount(&number_of_devices));
	LF_CORE_INFO("\tNumber of Devices = {0}", number_of_devices);

	for (unsigned int i = 0; i < number_of_devices; ++i) {
		char name[256];
		OPTIX_CALL(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name));
		LF_CORE_INFO("\tDevice {0}: {1}", i, name);

		int compute_capability[2] = {0, 0};
		OPTIX_CALL(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(
			compute_capability), &compute_capability));
		LF_CORE_INFO("\t\tCompute Support: {0}.{1}", compute_capability[0], compute_capability[1]);
	}
}
