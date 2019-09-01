#include "pch.h"
#include "OpenGLContext.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

lift::OpenGLContext::OpenGLContext(GLFWwindow* window_handle)
	: window_handle_(window_handle) {
	LF_CORE_ASSERT(window_handle_, "Window handle is null");
}


void lift::OpenGLContext::Init() {
	glfwMakeContextCurrent(window_handle_);
	const int status = gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
	LF_CORE_ASSERT(status, "Failed to initialize Glad");

	LF_CORE_INFO("");
	LF_CORE_INFO("OpenGL Info:");
	LF_CORE_INFO("	Vendor: {0}", glGetString(GL_VENDOR));
	LF_CORE_INFO("	Renderer: {0}", glGetString(GL_RENDERER));
	LF_CORE_INFO("	Version: {0}", glGetString(GL_VERSION));
}

void lift::OpenGLContext::SwapBuffers() {
	glfwSwapBuffers(window_handle_);
}
