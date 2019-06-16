#include "pch.h"
#include "Application.h"

#include "Log.h"

#include "ImGui/ImguiLayer.h"

//Temporary
#include <glad/glad.h>
#include "Renderer/Texture.h"
#include "Platform/OpenGL/OpenGLContext.h"

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


	void Application::InitOptix() {
	}

	void Application::InitGraphicsContext() {
		graphics_context_ = std::make_unique<OpenGLContext>(static_cast<GLFWwindow*>(window_->GetNativeWindow()));
		graphics_context_->Init();
	}

	void Application::CreateScene() {
		glGenVertexArrays(1, &vertex_array_);
		glBindVertexArray(vertex_array_);

		float vertices [4 * 9] = {
			-1.0f, -1.0f, 0.0f, 0.8f, 0.2f, 0.8f, 1.0f, 0.0f, 0.0f,
			1.0f, -1.0f, 0.0f, 0.2f, 0.3f, 0.8f, 1.0f, 1.0f, 0.0f,
			1.0f, 1.0f, 0.0f, 0.8f, 0.8f, 0.2f, 1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f, 0.0f, 0.1f, 0.8f, 0.2f, 1.0f, 0.0f, 1.0f,
		};

		vertex_buffer_.reset(VertexBuffer::Create(vertices, sizeof(vertices)));

		vertex_buffer_->SetLayout({
			{ShaderDataType::Float3, "a_Position"},
			{ShaderDataType::Float4, "a_Color"},
			{ShaderDataType::Float2, "a_Uv"}
		});

		uint32_t indices[6] = {0, 1, 2, 0, 2, 3};
		index_buffer_.reset(IndexBuffer::Create(indices, sizeof(indices) / sizeof(uint32_t)));

		Texture texture("res/textures/test.png");
		texture.Bind();
		shader_ = std::make_unique<Shader>("res/shaders/default");
		shader_->Bind();
		shader_->SetUniform1i("u_Texture", 0);
	}

	void Application::Render() {
		shader_->Bind();
		glBindVertexArray(vertex_array_);
		glDrawElements(GL_TRIANGLES, index_buffer_->GetCount(), GL_UNSIGNED_INT, nullptr);
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
}
