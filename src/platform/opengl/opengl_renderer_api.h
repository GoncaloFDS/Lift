#pragma once

#include "renderer/renderer_api.h"
#include <glm/common.hpp>

namespace lift {
class OpenGLRendererAPI : public RendererApi {
public:
    void setClearColor(const vec4& color) override;
    void clear() override;

    void drawIndexed(const std::shared_ptr<VertexArray>& vertex_array) override;
    void resize(uint32_t width, uint32_t height) override;
};
}
