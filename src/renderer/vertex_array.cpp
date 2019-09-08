#include "pch.h"
#include "vertex_array.h"
#include "renderer.h"
#include "platform/opengl/opengl_vertex_array.h"

lift::VertexArray* lift::VertexArray::create() {
    switch (Renderer::getApi()) {
        case RendererApi::API::NONE: LF_ASSERT(false, "RendererAPI::API::kNone is set");
            return nullptr;
        case RendererApi::API::OPEN_GL:
            return new OpenGLVertexArray();
    }
    LF_ASSERT(false, "Unkown RendererAPI");
    return nullptr;
}
