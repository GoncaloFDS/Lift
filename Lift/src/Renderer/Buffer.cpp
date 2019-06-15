#include "pch.h"
#include "Buffer.h"

#include "Renderer.h"
#include "Platform/OpenGL/OpenGLContext.h"
#include "Platform/OpenGL/OpenGLBuffer.h"


namespace lift {

	VertexBuffer* VertexBuffer::Create(float* vertices, const uint32_t size) {
		switch (Renderer::GetAPI()) {
			case RendererAPI::kNone:
			LF_CORE_ASSERT(false, "RendererAPI::kNone is set");
				return nullptr;
			case RendererAPI::kOpenGL:
				return new OpenGLVertexBuffer(vertices, size);
		}

		LF_CORE_ASSERT(false, "Unkown RenderAPI");
		return nullptr;
	}

	IndexBuffer* IndexBuffer::Create(uint32_t* indices, const uint32_t count) {

		switch (Renderer::GetAPI()) {
			case RendererAPI::kNone:
			LF_CORE_ASSERT(false, "RendererAPI::kNone is set");
				return nullptr;
			case RendererAPI::kOpenGL:
				return new OpenGLIndexBuffer(indices, count);
		}

		LF_CORE_ASSERT(false, "Unkown RenderAPI");
		return nullptr;
	}
}
