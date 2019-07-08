#pragma once
#include "RendererAPI.h"

namespace lift {
	class Renderer {
	public:
		static void BeginScene();
		static void EndScene();

		static void Submit(const std::shared_ptr<VertexArray>& vertex_array);

		static RendererAPI::API GetAPI() { return RendererAPI::GetAPI(); }

	};
}
