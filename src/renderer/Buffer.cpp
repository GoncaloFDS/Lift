#include "pch.h"
#include "Buffer.h"

#include "Renderer.h"
#include "platform/opengl/OpenGLBuffer.h"


lift::VertexBuffer* lift::VertexBuffer::Create(float* vertices, const uint32_t size) {
	switch (Renderer::GetAPI()) {
	case RendererAPI::API::None:
	LF_CORE_ASSERT(false, "RendererAPI::API::None is set");
		return nullptr;
	case RendererAPI::API::OpenGL:
		return new OpenGLVertexBuffer(vertices, size);
	}

	LF_CORE_ASSERT(false, "Unkown RenderAPI");
	return nullptr;
}

lift::IndexBuffer* lift::IndexBuffer::Create(uint32_t* indices, const uint32_t count) {

	switch (Renderer::GetAPI()) {
	case RendererAPI::API::None:
	LF_CORE_ASSERT(false, "RendererAPI::API::None is set");
		return nullptr;
	case RendererAPI::API::OpenGL:
		return new OpenGLIndexBuffer(indices, count);
	}

	LF_CORE_ASSERT(false, "Unkown RenderAPI");
	return nullptr;
}
