#pragma once

#include "renderer/renderer_api.h"
#include <glm/common.hpp>

namespace lift {

enum class BufferImageFormat {
    UNSIGNED_BYTE_4,
    FLOAT_4,
    FLOAT_3
};

class OpenGLRendererAPI : public RendererApi {
 public:
    void setClearColor(const vec4& color) override;
    void clear() override;

    void drawIndexed(const std::shared_ptr<VertexArray>& vertex_array) override;
    void resize(uint32_t width, uint32_t height) override;
};

auto pixelFormatSize(BufferImageFormat format) -> size_t;

}
