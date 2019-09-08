#pragma once

#include "renderer/vertex_array.h"

namespace lift {

class OpenGLVertexArray : public VertexArray {
public:

    OpenGLVertexArray();
    virtual ~OpenGLVertexArray() = default;

    void bind() const override;
    void unbind() const override;

    void addVertexBuffer(const std::shared_ptr<VertexBuffer> &vertex_buffer) override;
    void setIndexBuffer(const std::shared_ptr<IndexBuffer> &index_buffer) override;

    const std::vector<std::shared_ptr<VertexBuffer>> &getVertexBuffers() const override { return vertex_buffers_; }
    const std::shared_ptr<IndexBuffer> &getIndexBuffer() const override { return index_buffer_; }

private:
    uint32_t renderer_id_{};
    std::vector<std::shared_ptr<VertexBuffer>> vertex_buffers_{};
    std::shared_ptr<IndexBuffer> index_buffer_;
};
}
