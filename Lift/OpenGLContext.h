#pragma once

#include "Lift/Renderer/GraphicsContext.h"

struct GLFWWindow;

namespace lift {
	class OpenGLContext : public GraphicsContext {
	public:
		OpenGLContext(GLFWWindow* window_handle);

		void Init() override;
		void SwapBuffers() override;

	private:
		std::unique_ptr<GLFWWindow> window_handle_;
	};
}
