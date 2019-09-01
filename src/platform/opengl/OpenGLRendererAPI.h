#pragma once

#include "Renderer/RendererAPI.h"
#include <glm/common.hpp>

namespace lift {
	class OpenGLRendererAPI : public RendererAPI {
	public:
		void SetClearColor(const vec4& color) override;
		void Clear() override;


		void DrawIndexed(const std::shared_ptr<VertexArray>& vertex_array) override;
		void Resize(uint32_t width, uint32_t height) override;
	};
}
