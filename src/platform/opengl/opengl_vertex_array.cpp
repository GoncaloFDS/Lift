#include "pch.h"
#include "opengl_vertex_array.h"
#include "glad/glad.h"

static auto ShaderDataTypeToOpenGLBaseType(const lift::ShaderDataType type) -> GLenum {
    switch (type) {
        case lift::ShaderDataType::FLOAT:
        case lift::ShaderDataType::FLOAT_2:
        case lift::ShaderDataType::FLOAT_3:
        case lift::ShaderDataType::FLOAT_4:
        case lift::ShaderDataType::MAT_3:
        case lift::ShaderDataType::MAT_4:
            return GL_FLOAT;
        case lift::ShaderDataType::INT:
        case lift::ShaderDataType::INT_2:
        case lift::ShaderDataType::INT_3:
        case lift::ShaderDataType::INT_4:
            return GL_INT;
        case lift::ShaderDataType::Bool:
            return GL_BOOL;
        default: LF_ASSERT(false, "Unknown ShaderDataType");
            return 0;
    }
}

lift::OpenGLVertexArray::OpenGLVertexArray() {
    GL_CHECK(glCreateVertexArrays(1, &renderer_id_));
}

void lift::OpenGLVertexArray::bind() const {
    GL_CHECK(glBindVertexArray(renderer_id_));
}

void lift::OpenGLVertexArray::unbind() const {
    GL_CHECK(glBindVertexArray(0));
}

void lift::OpenGLVertexArray::addVertexBuffer(const std::shared_ptr<VertexBuffer>& vertex_buffer) {
    LF_ASSERT(vertex_buffer->getLayout().getElements().size(), "Vertex Buffer has no layout");

    GL_CHECK(glBindVertexArray(renderer_id_));
    vertex_buffer->bind();

    const auto& layout = vertex_buffer->getLayout();
    uint32_t index = 0;
    for (const auto& element : layout) {
        glEnableVertexAttribArray(index);
        glVertexAttribPointer(index,
                              element.getComponentCount(),
                              ShaderDataTypeToOpenGLBaseType(element.type),
                              element.normalized ? GL_TRUE : GL_FALSE,
                              layout.getStride(),
                              reinterpret_cast<const void*>(element.offset));
        index++;
    }

    vertex_buffers_.push_back(vertex_buffer);
}

void lift::OpenGLVertexArray::setIndexBuffer(const std::shared_ptr<IndexBuffer>& index_buffer) {
    GL_CHECK(glBindVertexArray(renderer_id_));
    index_buffer->bind();

    index_buffer_ = index_buffer;
}
