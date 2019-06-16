#include "pch.h"
#include "Application.h"

#include "Log.h"

#include "ImGui/ImguiLayer.h"

//Temporary
#include <glad/glad.h>
#include "Renderer/Texture.h"
#include "Platform/OpenGL/OpenGLContext.h"
#include "Platform/Optix/OptixErrorCodes.h"

namespace lift {

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

		while (is_running_) {
			StartFrame();

			Render();
			Display();

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
		std::shared_ptr<VertexBuffer> vertex_buffer {};
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

		Texture texture("res/textures/test.png");
		texture.Bind();
		shader_ = std::make_unique<Shader>("res/shaders/default");
		shader_->Bind();
		shader_->SetUniform1i("u_Texture", 0);
	}

	void Application::InitOptix() {
		GetOptixSystemInformation();

		optix_context_ = optix::Context::create();
		unsigned int device_count = 0;
		OPTIX_CALL(rtDeviceGetDeviceCount(&device_count));

		std::vector<int> devices;
		int devices_encoding = 3210;
		// Decimal digits encode OptiX device ordinals. Default 3210 means to use all four first installed devices, when available.
		unsigned int i = 0;
		do {
			int device = devices_encoding % 10;
			devices.push_back(device);
			devices_encoding /= 10;
			i++;
		} while (i < device_count && devices_encoding);

		optix_context_->setDevices(devices.begin(), devices.end());

		devices = optix_context_->getEnabledDevices();
		for (int device : devices) {
			LF_CORE_INFO("Optix context is using local device {0}: {1}", device, optix_context_->getDeviceName(device));
		}
	}

	void Application::InitGraphicsContext() {
		graphics_context_ = std::make_unique<OpenGLContext>(static_cast<GLFWwindow*>(window_->GetNativeWindow()));
		graphics_context_->Init();
	}

	void Application::Render() {
		shader_->Bind();
		vertex_array_->Bind();
		glDrawElements(GL_TRIANGLES, vertex_array_->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, nullptr);
	}

	void Application::Display() {
	}

	void Application::StartFrame() {
		glClearColor(0.1f, 0.1f, 0.1f, 1);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGuiLayer::Begin();
	}

	void Application::EndFrame() {
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
