#pragma once


#include "VertexArray.h"

namespace lift {
	class RendererAPI {
	public:
		enum class API {
			None = 0, OpenGL
		};

		virtual void SetClearColor(const vec4& color) = 0;
		virtual void Clear() = 0;
		virtual void Resize(uint32_t width, uint32_t height) = 0;

		virtual void DrawIndexed(const std::shared_ptr<VertexArray>& vertex_array) = 0;

		inline static API GetAPI() {return renderer_api_;}
	private:
		static API renderer_api_;
	};
}
