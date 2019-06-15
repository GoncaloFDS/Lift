#include "pch.h"
#include "OpenGLBuffer.h"

#include <glad/glad.h>

namespace lift {
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
			default: LF_CORE_ASSERT(false, "Unkown ShaderDataType");
				return 0;
		}

	}

	///
	/// VertexBuffer
	///

	OpenGLVertexBuffer::OpenGLVertexBuffer(float* vertices, const uint32_t size) {
		glCreateBuffers(1, &renderer_id_);
		glBindBuffer(GL_ARRAY_BUFFER, renderer_id_);
		glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);
	}

	OpenGLVertexBuffer::~OpenGLVertexBuffer() {
		glDeleteBuffers(1, &renderer_id_);
	}

	void OpenGLVertexBuffer::Bind() const {
		glBindBuffer(GL_ARRAY_BUFFER, renderer_id_);
	}

	void OpenGLVertexBuffer::Unbind() const {
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void OpenGLVertexBuffer::SetLayout(const BufferLayout& layout) {
		layout_ = layout;
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
	}

	///
	/// IndexBuffer
	/// 

	OpenGLIndexBuffer::OpenGLIndexBuffer(uint32_t* indices, const uint32_t count)
		: count_(count) {
		glCreateBuffers(1, &renderer_id_);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer_id_);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, count_ * sizeof(uint32_t), indices, GL_STATIC_DRAW);
	}

	OpenGLIndexBuffer::~OpenGLIndexBuffer() {
		glDeleteBuffers(1, &renderer_id_);
	}

	void OpenGLIndexBuffer::Bind() const {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer_id_);
	}

	void OpenGLIndexBuffer::Unbind() const {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	uint32_t OpenGLIndexBuffer::GetCount() const {
		return count_;
	}

}
