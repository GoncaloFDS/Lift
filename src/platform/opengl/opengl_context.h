#pragma once

#include "renderer/graphics_context.h"

struct GLFWwindow;

namespace lift {
class OpenGLContext : public GraphicsContext {
public:
    OpenGLContext(GLFWwindow* window_handle);

    void init() override;
    void swapBuffers() override;

private:
    GLFWwindow* window_handle_;
};
}
