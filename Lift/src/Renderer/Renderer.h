#pragma once

namespace lift {
	enum class RendererAPI {
		kNone = 0,
		kOpenGL = 1
	};

	class Renderer {
	public:
		static RendererAPI GetAPI() { return renderer_api_; }

	private:
		static RendererAPI renderer_api_;
	};
}
