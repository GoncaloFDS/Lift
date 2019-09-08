#include "pch.h"
#include "buffer.h"

#include "renderer.h"
#include "platform/opengl/opengl_buffer.h"

lift::VertexBuffer* lift::VertexBuffer::create(float* vertices, const uint32_t size) {
    switch (Renderer::getApi()) {
        case RendererApi::API::NONE: LF_ASSERT(false, "RendererAPI::API::None is set");
            return nullptr;
        case RendererApi::API::OPEN_GL:
            return new OpenGLVertexBuffer(vertices, size);
    }

    LF_ASSERT(false, "Unkown RenderAPI");
    return nullptr;
}

lift::IndexBuffer* lift::IndexBuffer::create(uint32_t* indices, const uint32_t count) {
    switch (Renderer::getApi()) {
        case RendererApi::API::NONE: LF_ASSERT(false, "RendererAPI::API::None is set");
            return nullptr;
        case RendererApi::API::OPEN_GL:
            return new OpenGLIndexBuffer(indices, count);
    }

    LF_ASSERT(false, "Unkown RenderAPI");
    return nullptr;
}
