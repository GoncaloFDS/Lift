#include "pch.h"
#include "RenderCommand.h"

#include "Platform/OpenGL/OpenGLRendererAPI.h"

namespace lift {
	RendererAPI* RenderCommand::renderer_api_ = new OpenGLRendererAPI();
}


