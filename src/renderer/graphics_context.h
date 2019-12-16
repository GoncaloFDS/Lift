#pragma once

namespace lift {
enum class BufferImageFormat {
    UNSIGNED_BYTE_4,
    FLOAT_4,
    FLOAT_3
};

class GraphicsContext {
public:
    virtual ~GraphicsContext() = default;
    virtual void init() = 0;
    virtual void swapBuffers() = 0;
};
}
