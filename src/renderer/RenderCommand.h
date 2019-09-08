#pragma once

#include "RendererAPI.h"

namespace lift {
class RenderCommand {
public:
    static void setClearColor(const vec4 &color) {
        renderer_api_->setClearColor(color);
    }

    static void clear() {
        renderer_api_->clear();
    }

    static void resize(uint32_t width, uint32_t height) {
        renderer_api_->resize(width, height);
    }

    static void drawIndexed(const std::shared_ptr<VertexArray> &vertex_array) {
        renderer_api_->drawIndexed(vertex_array);
    }

    static void shutdown() {
        delete renderer_api_;
    }

private:
    static RendererApi *renderer_api_;
};

}
