#pragma once

#include "RendererAPI.h"

namespace lift {
	class RenderCommand {
	public:
		inline static void SetClearColor(const mathfu::vec4& color) {
			renderer_api_->SetClearColor(color);
		}

		inline static void Clear() {
			renderer_api_->Clear();
		}

		static void DrawIndexed(const std::shared_ptr<VertexArray>& vertex_array) {
			renderer_api_->DrawIndexed(vertex_array);
		}
		
	private:
		static RendererAPI* renderer_api_;
	};


}
