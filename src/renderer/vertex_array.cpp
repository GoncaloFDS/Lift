#include "pch.h"
#include "vertex_array.h"
#include "renderer.h"
#include "platform/opengl/opengl_vertex_array.h"

auto lift::VertexArray::create() -> lift::VertexArray* {
    switch (Renderer::getApi()) {
        case RendererApi::API::NONE: LF_ASSERT(false, "RendererAPI::API::None is set");
            return nullptr;
        case RendererApi::API::OPEN_GL:
            return new OpenGLVertexArray();
    }
    LF_ASSERT(false, "Unkown RendererAPI");
    return nullptr;
}
