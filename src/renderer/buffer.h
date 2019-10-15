#pragma once

#include "pch.h"
#include <utility>

namespace lift {

enum class ShaderDataType {
    NONE = 0, FLOAT, FLOAT_2, FLOAT_3, FLOAT_4, MAT_3, MAT_4, INT, INT_2, INT_3, INT_4, Bool
};

static uint32_t shaderDataTypeSize(const ShaderDataType type) {
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
    ShaderDataType type;
    std::string name;
    uint32_t size;
    uint32_t offset;
    bool normalized;

    BufferElement() = default;

    BufferElement(const ShaderDataType type, const std::string& name, const bool normalized = false)
        : type{type}, name{std::move(name)}, size{shaderDataTypeSize(type)}, offset{0},
          normalized{normalized} {
    }

    uint32_t getComponentCount() const {
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

    [[nodiscard]] uint32_t getStride() const { return stride_; }
    [[nodiscard]] const std::vector<BufferElement>& getElements() const { return elements_; }

    std::vector<BufferElement>::iterator begin() { return elements_.begin(); }
    std::vector<BufferElement>::iterator end() { return elements_.end(); }
    [[nodiscard]] std::vector<BufferElement>::const_iterator begin() const { return elements_.begin(); }
    [[nodiscard]] std::vector<BufferElement>::const_iterator end() const { return elements_.end(); }

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

    [[nodiscard]] virtual const BufferLayout& getLayout() const = 0;
    virtual void setLayout(const BufferLayout& layout) = 0;

    static VertexBuffer* create(float* vertices, const uint32_t size);
};

class IndexBuffer {
public:
    virtual ~IndexBuffer() = default;

    virtual void bind() const = 0;
    virtual void unbind() const = 0;

    [[nodiscard]] virtual uint32_t getCount() const = 0;

    static IndexBuffer* create(uint32_t* indices, const uint32_t count);
};

}
