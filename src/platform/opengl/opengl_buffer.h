#pragma once

#include "renderer/buffer.h"

namespace lift {

class OpenGLVertexBuffer : public VertexBuffer {
public:
    OpenGLVertexBuffer(float* vertices, uint32_t size);
    virtual ~OpenGLVertexBuffer();

    void bind() const override;
    void unbind() const override;

    const BufferLayout& getLayout() const override { return layout_; }
    void setLayout(const BufferLayout& layout) override;

private:
    uint32_t renderer_id_{};
    BufferLayout layout_;
};

class OpenGLIndexBuffer : public IndexBuffer {
public:
    OpenGLIndexBuffer(uint32_t* indices, uint32_t count);
    virtual ~OpenGLIndexBuffer();

    void bind() const override;
    void unbind() const override;

    uint32_t getCount() const override;
private:
    uint32_t renderer_id_{};
    uint32_t count_;
};

}
