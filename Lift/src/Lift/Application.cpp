#include "pch.h"
#include "Application.h"

#include "Log.h"

#include <glad/glad.h>

//Temporary
#include <optix.h>
#include "optixu/optixpp_namespace.h"
#include "Platform/Optix/OptixErrorCodes.h"

namespace lift {

	Application* Application::instance_ = nullptr;

	static GLenum ShaderDataTypeToOpenGLBaseType(const ShaderDataType type) {
		switch (type) {
			case ShaderDataType::Float: return GL_FLOAT;
			case ShaderDataType::Float2: return GL_FLOAT;
			case ShaderDataType::Float3: return GL_FLOAT;
			case ShaderDataType::Float4: return GL_FLOAT;
			case ShaderDataType::Mat3: return GL_FLOAT;
			case ShaderDataType::Mat4: return GL_FLOAT;
			case ShaderDataType::Int: return GL_INT;
			case ShaderDataType::Int2: return GL_INT;
			case ShaderDataType::Int3: return GL_INT;
			case ShaderDataType::Int4: return GL_INT;
			case ShaderDataType::Bool: return GL_BOOL;
		}

		LF_CORE_ASSERT(false, "Unkown ShaderDataType");
		return 0;
	}

	Application::Application() {
		LF_CORE_ASSERT(!instance_, "Application already exists");
		instance_ = this;
		window_ = std::unique_ptr<Window>(Window::Create());
		window_->SetEventCallback(LF_BIND_EVENT_FN(Application::OnEvent));

		PushOverlay<ImguiLayer>();

		glGenVertexArrays(1, &vertex_array_);
		glBindVertexArray(vertex_array_);

		float vertices [4 * 7] = {
			-1.0f, -1.0f, 0.0f, 0.8f, 0.2f, 0.8f, 1.0f,
			1.0f, -1.0f, 0.0f, 0.2f, 0.3f, 0.8f, 1.0f,
			1.0f, 1.0f, 0.0f, 0.8f, 0.8f, 0.2f, 1.0f,
			-1.0f, 1.0f, 0.0f, 0.1f, 0.8f, 0.2f, 1.0f,
		};

		vertex_buffer_.reset(VertexBuffer::Create(vertices, sizeof(vertices)));

		{
			const BufferLayout layout = {
				{ShaderDataType::Float3, "a_Position"},
				{ShaderDataType::Float4, "a_Color"}
			};

			vertex_buffer_->SetLayout(layout);
		}

		uint32_t index = 0;
		const auto& layout = vertex_buffer_->GetLayout();
		for (const auto& element : layout) {
			glEnableVertexAttribArray(index);
			glVertexAttribPointer(index,
			                      element.GetComponentCount(),
			                      ShaderDataTypeToOpenGLBaseType(element.type),
			                      element.normalized ? GL_TRUE : GL_FALSE,
			                      layout.GetStride(),
			                      reinterpret_cast<const void*>(element.offset));
			index++;
		}

		uint32_t indices[6] = {0, 1, 2, 0, 2, 3};
		index_buffer_.reset(IndexBuffer::Create(indices, sizeof(indices) / sizeof(uint32_t)));

		const std::string vertex_src =
			R"(
			#version 330 core
			
			layout(location = 0) in vec3 a_Position;
			layout(location = 1) in vec4 a_Color;
		
			out vec3 v_Position;
			out vec4 v_Color;

			void main() {
				v_Position = a_Position;
				v_Color = a_Color;
				gl_Position = vec4(a_Position, 1.0);
			}
		)";

		const std::string fragment_src =
			R"(
			#version 330 core
			
			layout(location = 0) out vec4 color;

			in vec3 v_Position;
			in vec4 v_Color;

			void main()	{
				color = vec4(v_Position * 0.5 + 0.5, 1.0);
				color = v_Color;
			}
		)";

		shader_ = std::make_unique<Shader>(vertex_src, fragment_src);

		RTcontext context = nullptr;

		RTprogram ray_gen_program;
		RTbuffer buffer;

		RTvariable result_buffer;
		RTvariable draw_color;

		char path_to_ptx[512];
		char out_file[512];

		out_file[0] = '\0';
		RTresult result;
		OPTIX_CALL(rtContextCreate(&context));
		OPTIX_CALL(rtContextSetRayTypeCount(context, 1));
		OPTIX_CALL(rtContextSetEntryPointCount(context, 1));

		OPTIX_CALL(rtBufferCreate(context, RT_BUFFER_OUTPUT, &buffer));
		OPTIX_CALL(rtBufferSetFormat(buffer, RT_FORMAT_FLOAT4));
		OPTIX_CALL(rtBufferSetSize2D(buffer, window_->GetWidth(), window_->GetHeight()));
		OPTIX_CALL(rtContextDeclareVariable(context, "result_buffer", &result_buffer));
		OPTIX_CALL(rtVariableSetObject(result_buffer, buffer));

		sprintf(path_to_ptx, "%s/%s", "Resources", "optixHello_generated_draw_color.cu.ptx");
		OPTIX_CALL(rtProgramCreateFromPTXFile(context, path_to_ptx, "draw_solid_color", &ray_gen_program));
		OPTIX_CALL(rtProgramDeclareVariable(ray_gen_program, "draw_color", &draw_color));
		OPTIX_CALL(rtVariableSet3f(draw_color, 0.4f, 0.7, 0.0f));
		OPTIX_CALL(rtContextSetRayGenerationProgram(context, 0, ray_gen_program));

		// Run
		OPTIX_CALL(rtContextValidate(context));
		OPTIX_CALL(rtContextLaunch2D(context, 0, window_->GetWidth(), window_->GetHeight()));

		// Display image


		// Clean up
		OPTIX_CALL(rtBufferDestroy(buffer));
		OPTIX_CALL(rtProgramDestroy(ray_gen_program));
		OPTIX_CALL(rtContextDestroy(context));


	}

	void Application::Run() {


		while (is_running_) {
			glClearColor(0.1f, 0.1f, 0.1f, 1);
			glClear(GL_COLOR_BUFFER_BIT);

			shader_->Bind();
			glBindVertexArray(vertex_array_);
			glDrawElements(GL_TRIANGLES, index_buffer_->GetCount(), GL_UNSIGNED_INT, nullptr);

			for (auto& layer : layer_stack_)
				layer->OnUpdate();

			ImguiLayer::Begin();
			for (auto& layer : layer_stack_)
				layer->OnImguiRender();
			ImguiLayer::End();

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
