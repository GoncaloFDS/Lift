#pragma once
#include "buffer.h"

namespace lift {

class VertexArray {
public:
    VertexArray() = default;
    virtual ~VertexArray() = default;

    virtual void bind() const = 0;
    virtual void unbind() const = 0;

    virtual void addVertexBuffer(const std::shared_ptr<VertexBuffer>& vertex_buffer) = 0;
    virtual void setIndexBuffer(const std::shared_ptr<IndexBuffer>& index_buffer) = 0;

    [[nodiscard]] virtual auto getVertexBuffers() const -> const std::vector<std::shared_ptr<VertexBuffer>>& = 0;
    [[nodiscard]] virtual auto getIndexBuffer() const -> const std::shared_ptr<IndexBuffer>& = 0;

    static auto create() -> VertexArray*;

};
}
