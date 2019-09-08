#pragma once

namespace lift {

struct PixelBuffer {
    PixelBuffer(float size);
    virtual ~PixelBuffer() = default;

    virtual void bind() const;
    virtual void unbind() const;

    virtual void resize(const unsigned size);

    unsigned id{};
};
}
