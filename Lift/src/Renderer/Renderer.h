#pragma once
#include "RendererAPI.h"
#include "CudaBuffer.h"
#include <cuda/launch_parameters.h>


namespace lift {
	class Scene;

	class Renderer {
	public:
		Renderer();

		void LaunchSubframe(const Scene &scene, LaunchParameters &params, const ivec2 &size);
		static void Submit(const std::shared_ptr<VertexArray>& vertex_array);
		static RendererAPI::API GetAPI() { return RendererAPI::GetAPI(); }

	private:
		
		CudaBuffer<LaunchParameters> d_params_;
	};
}
