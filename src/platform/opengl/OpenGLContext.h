#pragma once

#include "Renderer/GraphicsContext.h"

struct GLFWwindow;

namespace lift {
	class OpenGLContext : public GraphicsContext {
	public:
		OpenGLContext(GLFWwindow* window_handle);

		void Init() override;
		void SwapBuffers() override;

	private:
		GLFWwindow* window_handle_;
	};
}
