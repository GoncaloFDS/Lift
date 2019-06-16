#include "pch.h"
#include "OpenGLVertexArray.h"
#include "glad/glad.h"

namespace lift {


	static GLenum ShaderDataTypeToOpenGLBaseType(const lift::ShaderDataType type) {
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
			default: LF_CORE_ASSERT(false, "Unkown ShaderDataType");
				return 0;
		}
	}

	OpenGLVertexArray::OpenGLVertexArray() {
		OPENGL_CALL(glCreateVertexArrays(1, &renderer_id_));
	}

	void OpenGLVertexArray::Bind() const {
		OPENGL_CALL(glBindVertexArray(renderer_id_));
	}

	void OpenGLVertexArray::Unbind() const {
		OPENGL_CALL(glBindVertexArray(0));
	}

	void OpenGLVertexArray::AddVertexBuffer(const std::shared_ptr<VertexBuffer>& vertex_buffer) {
		LF_CORE_ASSERT(vertex_buffer->GetLayout().GetElements().size(), "Vertex Buffer has no layout");

		OPENGL_CALL(glBindVertexArray(renderer_id_));
		vertex_buffer->Bind();

		const auto& layout = vertex_buffer->GetLayout();
		uint32_t index = 0;
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

		vertex_buffers_.push_back(vertex_buffer);
	}

	void OpenGLVertexArray::SetIndexBuffer(const std::shared_ptr<IndexBuffer>& index_buffer) {
		OPENGL_CALL(glBindVertexArray(renderer_id_));
		index_buffer->Bind();

		index_buffer_ = index_buffer;
	}

}
