#include "pch.h"
#include "VertexArray.h"
#include "Renderer.h"
#include "Platform/OpenGL/OpenGLVertexArray.h"

namespace lift {

	VertexArray* VertexArray::Create() {
		switch (Renderer::GetAPI()) {
			case RendererAPI::kNone:
			LF_CORE_ASSERT(false, "RendererAPI::kNone is set");
				return nullptr;
			case RendererAPI::kOpenGL:
				return new OpenGLVertexArray();
		}
		LF_CORE_ASSERT(false, "Unkown RendererAPI");
		return nullptr;
	}
}
