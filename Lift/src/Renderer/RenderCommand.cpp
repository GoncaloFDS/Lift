#include "pch.h"
#include "RenderCommand.h"

#include "Platform/OpenGL/OpenGLRendererAPI.h"

lift::RendererAPI* lift::RenderCommand::renderer_api_ = new lift::OpenGLRendererAPI();
