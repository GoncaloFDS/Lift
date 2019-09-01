#include "pch.h"
#include "Renderer.h"
#include "RenderCommand.h"

#include <optix.h>
#include <optix_stubs.h>
#include "scene/Scene.h"
#include "cuda/launch_parameters.h"

lift::Renderer::Renderer() {
	d_params_.alloc(1);
}


void lift::Renderer::LaunchSubframe(const Scene& scene, LaunchParameters& params, const ivec2& size) {
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params_.get()),
		&params,
		sizeof(LaunchParameters),
		cudaMemcpyHostToDevice,
		nullptr));
	
	OPTIX_CHECK( optixLaunch(
		scene.GetPipeline(),
		nullptr,
		d_params_.get(),
		sizeof(LaunchParameters),
		scene.GetSbt(),
		size.x,
		size.y,
		1));
	
	CUDA_SYNC_CHECK();
}

void lift::Renderer::Submit(const std::shared_ptr<VertexArray>& vertex_array) {
	vertex_array->Bind();
	RenderCommand::DrawIndexed(vertex_array);
	
}
