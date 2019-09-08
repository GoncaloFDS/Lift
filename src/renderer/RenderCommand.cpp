#include "pch.h"
#include "RenderCommand.h"

#include "platform/opengl/OpenGLRendererAPI.h"

lift::RendererApi *lift::RenderCommand::renderer_api_ = new lift::OpenGLRendererAPI();
