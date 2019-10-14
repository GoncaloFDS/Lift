#include "pch.h"
#include "opengl_renderer_api.h"

#include <glad/glad.h>

void lift::OpenGLRendererAPI::setClearColor(const vec4& color) {
    glClearColor(color.x, color.y, color.z, color.w);
}

void lift::OpenGLRendererAPI::clear() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void lift::OpenGLRendererAPI::drawIndexed(const std::shared_ptr<VertexArray>& vertex_array) {
    glDrawElements(GL_TRIANGLES, vertex_array->getIndexBuffer()->getCount(), GL_UNSIGNED_INT, nullptr);
}

void lift::OpenGLRendererAPI::resize(uint32_t width, uint32_t height) {
    glViewport(0, 0, width, height);
}

size_t lift::pixelFormatSize(lift::BufferImageFormat format) {
    switch (format) {
        case BufferImageFormat::UNSIGNED_BYTE_4:
            return sizeof(char) * 4;
        case BufferImageFormat::FLOAT_3:
            return sizeof(float) * 3;
        case BufferImageFormat::FLOAT_4:
            return sizeof(float) * 4;
    }
    return -1;
}
