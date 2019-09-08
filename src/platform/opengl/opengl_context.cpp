#include "pch.h"
#include "opengl_context.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

lift::OpenGLContext::OpenGLContext(GLFWwindow *window_handle)
    : window_handle_(window_handle) {
    LF_ASSERT(window_handle_, "Window handle is null");
}

void lift::OpenGLContext::init() {
    glfwMakeContextCurrent(window_handle_);
    const int status = gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
    LF_ASSERT(status, "Failed to initialize Glad");

    LF_INFO("");
    LF_INFO("OpenGL Info:");
    LF_INFO("	Vendor: {0}", glGetString(GL_VENDOR));
    LF_INFO("	Renderer: {0}", glGetString(GL_RENDERER));
    LF_INFO("	Version: {0}", glGetString(GL_VERSION));
}

void lift::OpenGLContext::swapBuffers() {
    glfwSwapBuffers(window_handle_);
}
