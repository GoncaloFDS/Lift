#pragma once

#include "Renderer/GraphicsContext.h"

struct GLFWwindow;

namespace lift {
class OpenGLContext : public GraphicsContext {
public:
    OpenGLContext(GLFWwindow *window_handle);

    void init() override;
    void swapBuffers() override;

private:
    GLFWwindow *window_handle_;
};
}
