#include "pch.h"
#include "Renderer.h"
#include "RenderCommand.h"

#include <optix.h>
#include <optix_stubs.h>
#include "scene/Scene.h"
#include "cuda/launch_parameters.cuh"

lift::Renderer::Renderer() {
	d_params_.alloc(sizeof(LaunchParameters));
}


void lift::Renderer::LaunchSubframe(const Scene& scene, LaunchParameters& params) {
	params_ = params;
	
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
		params.frame.size.x,
		params.frame.size.y,
		1));
	
	CUDA_SYNC_CHECK();
}

void lift::Renderer::DownloadFrame(uint32_t pixels[], CudaBuffer<uint32_t> buffer) {
	buffer.download(pixels);
}

void lift::Renderer::Submit(const std::shared_ptr<VertexArray>& vertex_array) {
	vertex_array->Bind();
	RenderCommand::DrawIndexed(vertex_array);
	
}
