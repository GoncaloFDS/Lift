#include "pch.h"
#include "Application.h"

#include "Log.h"

#include "ImGui/ImguiLayer.h"

//Temporary
#include <glad/glad.h>
#include "Renderer/Texture.h"
#include "Platform/OpenGL/OpenGLContext.h"
#include "Platform/Optix/OptixErrorCodes.h"
#include "Renderer/Renderer.h"
#include "Renderer/RenderCommand.h"
#include "Cuda/ParallelogramLight.cuh"
#include <fstream>

namespace lift {
	////////////////////////////////////////////////////

	std::string GetPtxString(const char* file_name) {
		std::string ptx_source;

		const std::ifstream file(file_name);
		if (file.good()) {
			std::stringstream source_buffer;
			source_buffer << file.rdbuf();
			return source_buffer.str();
		}
		LF_CORE_ERROR("Invalid PTX path: {0}", file_name);
		return "Invalid PTX";
	}

	//////////////////////////////////////////////////
	Application* Application::instance_ = nullptr;

	Application::Application() {
		LF_CORE_ASSERT(!instance_, "Application already exists");
		instance_ = this;
		window_ = std::unique_ptr<Window>(Window::Create());
		window_->SetEventCallback(LF_BIND_EVENT_FN(Application::OnEvent));
		//window_->SetVSync(false);

		InitGraphicsContext();
		InitOptix();

		PushOverlay<ImGuiLayer>();
	}

	void Application::Run() {
		CreateScene();
		optix_context_->validate();

		while (is_running_) {
			ImGuiLayer::Begin();

			RenderCommand::SetClearColor({0.1f, 0.1f, 0.1f, 0.0f});
			RenderCommand::Clear();
			
			optix_context_->launch(0, window_->GetWidth(), window_->GetHeight());

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, hdr_texture_);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_output_->getGLBOId());
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, window_->GetWidth(), window_->GetHeight(), 0, GL_RGBA, GL_FLOAT, nullptr);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, hdr_texture_);

			shader_->Bind();
			Renderer::Submit(vertex_array_);

			for (auto& layer : layer_stack_)
				layer->OnUpdate();

			for (auto& layer : layer_stack_)
				layer->OnImguiRender();
			EndFrame();
		}
	}

	void Application::CreateScene() {

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

		shader_ = std::make_unique<Shader>("res/shaders/default");
		shader_->Bind();
		shader_->SetUniform1i("u_Texture", 0);

	}

	void Application::InitOptix() {
		GetOptixSystemInformation();

		optix_context_ = optix::Context::create();

		InitPrograms();

		optix_context_->setRayTypeCount(0);
		optix_context_->setEntryPointCount(1);
		optix_context_->setStackSize(1800);
		optix_context_->setMaxTraceDepth(2);
		optix_context_->setPrintEnabled(true);
		optix_context_->setExceptionEnabled(RT_EXCEPTION_ALL, true);

		optix_context_["sysColorBackground"]->setFloat(0.46f, 0.72f, 0.0f);

		buffer_output_ = optix_context_->createBufferFromGLBO(RT_BUFFER_OUTPUT, hdr_texture_);
		buffer_output_->setFormat(RT_FORMAT_FLOAT4); //RGBA32F
		buffer_output_->setSize(window_->GetWidth(), window_->GetHeight());
		optix_context_["sysOutputBuffer"]->set(buffer_output_);

		auto it = ptx_programs_.find("raygeneration");
		optix_context_->setRayGenerationProgram(0, it->second);

		it = ptx_programs_.find("exception");
		optix_context_->setExceptionProgram(0, it->second);

	}

	void Application::InitGraphicsContext() {
		graphics_context_ = std::make_unique<OpenGLContext>(static_cast<GLFWwindow*>(window_->GetNativeWindow()));
		graphics_context_->Init();

		glGenBuffers(1, &pbo_output_buffer_);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_output_buffer_);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, window_->GetWidth() * window_->GetHeight() * sizeof(float) * 4, nullptr,
					 GL_STREAM_READ);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		///////////////

		glGenTextures(1, &hdr_texture_);
		glBindTexture(GL_TEXTURE_2D, hdr_texture_);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, 0);

	}

	void Application::InitPrograms() {
		std::string ptx_string = GetPtxString("res/ptx/raygeneration.ptx");
		ptx_programs_["raygeneration"] = optix_context_->createProgramFromPTXString(ptx_string, "raygeneration");

		ptx_string = GetPtxString("res/ptx/exception.ptx");
		ptx_programs_["exception"] = optix_context_->createProgramFromPTXString(ptx_string, "exception");
	}

	void Application::EndFrame() const {
		ImGuiLayer::End();
		graphics_context_->SwapBuffers();
		window_->OnUpdate();
	}

	void Application::OnEvent(Event& e) {
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<WindowCloseEvent>(LF_BIND_EVENT_FN(Application::OnWindowClose));

		for (auto it = layer_stack_.end(); it != layer_stack_.begin();) {
			(*--it)->OnEvent(e);
			if (e.handled_)
				break;
		}
	}

	bool Application::OnWindowClose(WindowCloseEvent& e) {
		is_running_ = false;

		return true;
	}

	void Application::GetOptixSystemInformation() {
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
}
