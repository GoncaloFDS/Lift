#include "pch.h"
#include "render_command.h"

#include "platform/opengl/opengl_renderer_api.h"

lift::RendererApi *lift::RenderCommand::renderer_api_ = new lift::OpenGLRendererAPI();
