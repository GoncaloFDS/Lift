#pragma once
#include "cuda_output_buffer.h"
#include "optix_context.h"
#include <cuda/launch_parameters.h>
#include <GLFW/glfw3.h>
#include <scene/cameras/camera.h>
#include <platform/opengl/opengl_context.h>

namespace lift {
class Scene;
class Window;

class Renderer {
 public:
    void init(CudaOutputBufferType type, const std::shared_ptr<Window>& window);

    void initOptixContext(const Scene& scene);

    void launchSubframe(const Scene& scene, const ivec2& size);

    void displaySubframe(void* window);

    void onResize(int32_t width, int32_t height);

    void allocLights(Scene& scene);

    void updateLaunchParameters(Scene scene);

	void setClearColor(const vec3& color);
    auto clearColor() -> vec3;
    void resetFrame();
 private:
    void createOutputBuffer(CudaOutputBufferType type, ivec2 frame_size);
    void resizeOutputBuffer(int32_t width, int32_t height);
    void resizeAccumulationButter(int32_t width, int32_t height);

private:
	OptixContext optix_context_;
	std::unique_ptr<OpenGLContext> graphics_context_;
	std::unique_ptr<CudaOutputBuffer<uchar4>> output_buffer_;
    LaunchParameters launch_parameters_;
    LaunchParameters* d_params_;

    vec3 clear_color_;

};
}
