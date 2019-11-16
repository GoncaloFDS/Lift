#pragma once

#include "renderer/buffer.h"

namespace lift {

class OpenGLVertexBuffer : public VertexBuffer {
public:
    OpenGLVertexBuffer(float* vertices, uint32_t size);
    ~OpenGLVertexBuffer() override;

    void bind() const override;
    void unbind() const override;

    [[nodiscard]] auto getLayout() const -> const BufferLayout& override { return layout_; }
    void setLayout(const BufferLayout& layout) override;

private:
    uint32_t renderer_id_{};
    BufferLayout layout_;
};

class OpenGLIndexBuffer : public IndexBuffer {
public:
    OpenGLIndexBuffer(uint32_t* indices, uint32_t count);
    ~OpenGLIndexBuffer() override;

    void bind() const override;
    void unbind() const override;

    [[nodiscard]] auto getCount() const -> uint32_t override;
private:
    uint32_t renderer_id_{};
    uint32_t count_;
};

}
