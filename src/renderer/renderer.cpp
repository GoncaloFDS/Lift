#include "pch.h"
#include "renderer.h"

#include <optix.h>
#include <optix_stubs.h>
#include <core/profiler.h>
#include <GLFW/glfw3.h>
#include <cuda/launch_parameters.h>
#include <cuda/math_constructors.h>
#include <cuda/vec_math.h>
#include "scene/scene.h"
#include "cuda_buffer.h"


void lift::Renderer::init(CudaOutputBufferType type, const std::shared_ptr<Window>& window) {
	LF_ASSERT(type == CudaOutputBufferType::GL_INTEROP, "Unsupported CudaOutputBufferType");
	graphics_context_ = std::make_unique<OpenGLContext>(window);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &d_params_ ), sizeof(LaunchParameters)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &launch_parameters_.accum_buffer ),
						  window->size().x * window->size().y * sizeof(float4)));
	launch_parameters_.frame_buffer = nullptr;
	launch_parameters_.subframe_index = 0u;
	launch_parameters_.samples_per_launch = 1;
	launch_parameters_.max_depth = 3;
	launch_parameters_.width = window->width();
	launch_parameters_.height = window->height();
	setClearColor(vec3(0.1f));
	createOutputBuffer(type, window->size());
}

void lift::Renderer::launchSubframe(const Scene& scene, const ivec2& size) {
	if (size.x == 0 || size.y == 0)
		return;
	Profiler profiler(Profiler::Id::Render);
	uchar4* result_buffer_data = output_buffer_->map();
	launch_parameters_.frame_buffer = result_buffer_data;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params_),
							   &launch_parameters_,
							   sizeof(LaunchParameters),
							   cudaMemcpyHostToDevice,
							   nullptr));

	OPTIX_CHECK(optixLaunch(
		optix_context_.pipeline(),
		nullptr,
		reinterpret_cast<CUdeviceptr>( d_params_ ),
		sizeof(LaunchParameters),
		optix_context_.sbt(),
		size.x,
		size.y,
		1));

	output_buffer_->unmap();
	CUDA_SYNC_CHECK();

	launch_parameters_.subframe_index++;
}

void lift::Renderer::displaySubframe(void* window) {
	Profiler profiler(Profiler::Id::Display);
	int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
	int framebuf_res_y = 0;   //
	glfwGetFramebufferSize(static_cast<GLFWwindow*>(window), &framebuf_res_x, &framebuf_res_y);
	graphics_context_->display(
		ivec2(output_buffer_->width(), output_buffer_->height()),
		ivec2(framebuf_res_x, framebuf_res_y),
		output_buffer_->getPixelBufferObject()
	);
}

void lift::Renderer::createOutputBuffer(CudaOutputBufferType type, ivec2 frame_size) {
	output_buffer_ = std::make_unique<CudaOutputBuffer<uchar4>>(type, frame_size.x, frame_size.y);

}

void lift::Renderer::allocLights(Scene& scene) {
	auto& lights = scene.lights();
	launch_parameters_.lights.count = static_cast<uint32_t>(lights.size());
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&launch_parameters_.lights.data),
		lights.size() * sizeof(Light)

	));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>( launch_parameters_.lights.data ),
		lights.data(),
		lights.size() * sizeof(Light),
		cudaMemcpyHostToDevice
	));
}

void lift::Renderer::updateLaunchParameters(Scene scene) {
	auto camera = scene.camera();

	launch_parameters_.camera.eye = makeFloat3(camera->eye());
	launch_parameters_.camera.u = makeFloat3(camera->vectorU());
	launch_parameters_.camera.v = makeFloat3(camera->vectorV());
	launch_parameters_.camera.w = makeFloat3(camera->vectorW());
	launch_parameters_.handle = optix_context_.traversableHandle();
}

void lift::Renderer::onResize(int32_t width, int32_t height) {
	resizeOutputBuffer(width, height);
	resizeAccumulationButter(width, height);
	launch_parameters_.width = width;
	launch_parameters_.height = height;
	resetFrame();
}

void lift::Renderer::resizeOutputBuffer(int32_t width, int32_t height) {
	output_buffer_->resize(width, height);
}

void lift::Renderer::resizeAccumulationButter(int32_t width, int32_t height) {
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>( launch_parameters_.accum_buffer )));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &launch_parameters_.accum_buffer ),
						  width * height * sizeof(float4)
	));
}

void lift::Renderer::setClearColor(const vec3& color) {
	clear_color_ = color;

	//launch_parameters_.miss_color = makeFloat3(clear_color_);
}

auto lift::Renderer::clearColor() -> vec3 {
	return clear_color_;
}

void lift::Renderer::resetFrame() {
	launch_parameters_.subframe_index = 0u;
}

void lift::Renderer::initOptixContext(const lift::Scene& scene) {
	optix_context_.init(scene);
}

