#include "pch.h"
#include "VertexArray.h"
#include "Renderer.h"
#include "Platform/OpenGL/OpenGLVertexArray.h"

lift::VertexArray* lift::VertexArray::Create() {
	switch (Renderer::GetAPI()) {
	case RendererAPI::API::None:
	LF_CORE_ASSERT(false, "RendererAPI::API::kNone is set");
		return nullptr;
	case RendererAPI::API::OpenGL:
		return new OpenGLVertexArray();
	}
	LF_CORE_ASSERT(false, "Unkown RendererAPI");
	return nullptr;
}
