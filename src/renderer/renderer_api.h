#pragma once

#include "vertex_array.h"

namespace lift {
class RendererApi {
public:
    enum class API {
        NONE = 0,
        OPEN_GL
    };

    virtual ~RendererApi() = default;

    virtual void setClearColor(const vec4& color) = 0;
    virtual void clear() = 0;
    virtual void resize(uint32_t width, uint32_t height) = 0;

    virtual void drawIndexed(const std::shared_ptr<VertexArray>& vertex_array) = 0;

    inline static auto getApi() -> API { return renderer_api_; }
private:
    static API renderer_api_;
};
}
