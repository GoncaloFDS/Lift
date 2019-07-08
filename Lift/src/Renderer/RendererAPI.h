#pragma once


#include "mathfu/vector.h"
#include "mathfu/glsl_mappings.h"
#include "VertexArray.h"

namespace lift {
	class RendererAPI {
	public:
		enum class API {
			None = 0, OpenGL
		};

		virtual void SetClearColor(const mathfu::vec4& color ) = 0;
		virtual void Clear() = 0;

		virtual void DrawIndexed(const std::shared_ptr<VertexArray>& vertex_array) = 0;

		inline static API GetAPI() {return renderer_api_;}
	private:
		static API renderer_api_;
	};
}
