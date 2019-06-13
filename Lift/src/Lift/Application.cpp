#include "pch.h"
#include "Application.h"

#include "Log.h"

#include <glad/glad.h>

//Temporary
#include <optix.h>
#include "optixu/optixpp_namespace.h"

namespace lift {


	Application* Application::instance_ = nullptr;

	Application::Application() {
		LF_CORE_ASSERT(!instance_, "Application already exists");
		instance_ = this;
		window_ = std::unique_ptr<Window>(Window::Create());
		window_->SetEventCallback(LF_BIND_EVENT_FN(Application::OnEvent));

		PushOverlay<ImGuiLayer>();

		glGenVertexArrays(1, &vertex_array_);
		glBindVertexArray(vertex_array_);

		glGenBuffers(1, &vertex_buffer_);
		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);

		float vertices [3 * 3] = {
			-0.5f, -0.5f, 0.0f,
			0.5f, -0.5f, 0.0f,
			0.0f, 0.5f, 0.0f,
		};

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

		glGenBuffers(1, &index_buffer_);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);

		unsigned int indices[3] = {0, 1, 2};
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		const std::string vertex_src =
			R"(
			#version 330 core
			
			layout(location = 0) in vec3 a_Position;
			out vec3 v_Position;
			void main() {
				v_Position = a_Position * 0.1;
				gl_Position = vec4(v_Position, 1.0);	
			}
		)";

		const std::string fragment_src =
			R"(
			#version 330 core
			
			layout(location = 0) out vec4 color;
			in vec3 v_Position;
			void main()	{
				color = vec4(v_Position * 0.5 + 0.5, 1.0);
			}
		)";

		shader_ = std::make_unique<Shader>(vertex_src, fragment_src);

	}

	void Application::Run() {

		RTprogram ray_gen_program;
		RTbuffer buffer;

		while (is_running_) {
			glClearColor(0.1f, 0.1f, 0.1f, 1);
			glClear(GL_COLOR_BUFFER_BIT);

			shader_->Bind();
			glBindVertexArray(vertex_array_);
			glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, nullptr);

			for (auto& layer : layer_stack_)
				layer->OnUpdate();

			ImGuiLayer::Begin();
			for (auto& layer : layer_stack_)
				layer->OnImGuiRender();
			ImGuiLayer::End();

			window_->OnUpdate();
		}
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
