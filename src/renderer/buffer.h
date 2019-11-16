#pragma once

#include "pch.h"
#include <utility>

namespace lift {

enum class ShaderDataType {
    NONE = 0, FLOAT, FLOAT_2, FLOAT_3, FLOAT_4, MAT_3, MAT_4, INT, INT_2, INT_3, INT_4, Bool
};

static auto shaderDataTypeSize(const ShaderDataType type) -> uint32_t {
    switch (type) {
        case ShaderDataType::NONE:
            break;
        case ShaderDataType::FLOAT:
            return 4;
        case ShaderDataType::FLOAT_2:
            return 4 * 2;
        case ShaderDataType::FLOAT_3:
            return 4 * 3;
        case ShaderDataType::FLOAT_4:
            return 4 * 4;
        case ShaderDataType::MAT_3:
            return 4 * 3 * 3;
        case ShaderDataType::MAT_4:
            return 4 * 4 * 4;
        case ShaderDataType::INT:
            return 4;
        case ShaderDataType::INT_2:
            return 4 * 2;
        case ShaderDataType::INT_3:
            return 4 * 3;
        case ShaderDataType::INT_4:
            return 4 * 4;
        case ShaderDataType::Bool:
            return 1;
    }

    return 0;
}

struct BufferElement {
    ShaderDataType type = ShaderDataType::NONE;
    std::string name;
    uint32_t size{};
    uint32_t offset{};
    bool normalized{};

    BufferElement() = default;

    BufferElement(const ShaderDataType type, std::string name, const bool normalized = false)
        : type{type}, name{std::move(name)}, size{shaderDataTypeSize(type)}, offset{0},
          normalized{normalized} {
    }

    [[nodiscard]] auto getComponentCount() const -> uint32_t {
        switch (type) {
            case ShaderDataType::FLOAT:
                return 1;
            case ShaderDataType::FLOAT_2:
                return 2;
            case ShaderDataType::FLOAT_3:
                return 3;
            case ShaderDataType::FLOAT_4:
                return 4;
            case ShaderDataType::MAT_3:
                return 3 * 3;
            case ShaderDataType::MAT_4:
                return 4 * 4;
            case ShaderDataType::INT:
                return 1;
            case ShaderDataType::INT_2:
                return 2;
            case ShaderDataType::INT_3:
                return 3;
            case ShaderDataType::INT_4:
                return 4;
            case ShaderDataType::Bool:
                return 1;
            case ShaderDataType::NONE:
                LF_ASSERT(false, "Shader Data Type set to None");
                break;
        }

        return 0;
    }
};

class BufferLayout {
public:
    BufferLayout() = default;

    BufferLayout(const std::initializer_list<BufferElement>& elements)
        : elements_{elements} {
        calculateOffsetAndStride();
    }

    [[nodiscard]] auto getStride() const -> uint32_t { return stride_; }
    [[nodiscard]] auto getElements() const -> const std::vector<BufferElement>& { return elements_; }

    auto begin() -> std::vector<BufferElement>::iterator { return elements_.begin(); }
    auto end() -> std::vector<BufferElement>::iterator { return elements_.end(); }
    [[nodiscard]] auto begin() const -> std::vector<BufferElement>::const_iterator { return elements_.begin(); }
    [[nodiscard]] auto end() const -> std::vector<BufferElement>::const_iterator { return elements_.end(); }

private:
    void calculateOffsetAndStride() {
        uint32_t offset = 0;
        stride_ = 0;
        for (auto& element : elements_) {
            element.offset = offset;
            offset += element.size;
            stride_ += element.size;
        }
    }

private:
    std::vector<BufferElement> elements_;
    uint32_t stride_ = 0;
};

class VertexBuffer {
public:
    virtual ~VertexBuffer() = default;

    virtual void bind() const = 0;
    virtual void unbind() const = 0;

    [[nodiscard]] virtual auto getLayout() const -> const BufferLayout& = 0;
    virtual void setLayout(const BufferLayout& layout) = 0;

    static auto create(float* vertices, const uint32_t size) -> VertexBuffer*;
};

class IndexBuffer {
public:
    virtual ~IndexBuffer() = default;

    virtual void bind() const = 0;
    virtual void unbind() const = 0;

    [[nodiscard]] virtual auto getCount() const -> uint32_t = 0;

    static auto create(uint32_t* indices, const uint32_t count) -> IndexBuffer*;
};

}
