#pragma once
#include "renderer_api.h"
#include "cuda_output_buffer.h"
#include <cuda/launch_parameters.h>
#include <platform/opengl/opengl_display.h>
#include <GLFW/glfw3.h>
#include <scene/cameras/camera.h>

namespace lift {
class Scene;
class Window;

class Renderer {
 public:
    static RendererApi::API getApi() { return RendererApi::getApi(); }

    void init(CudaOutputBufferType type, ivec2 frame_size);

    void launchSubframe(const Scene& scene, const ivec2& size);

    void displaySubframe(OpenGLDisplay& gl_display, void* window);

    void onResize(int32_t width, int32_t height);

    void allocLights(Scene& scene);

    void updateLaunchParameters(Scene scene);

    void setClearColor(const vec3& color);
    vec3 clearColor();
    void resetFrame();
 private:
    void createOutputBuffer(CudaOutputBufferType type, ivec2 frame_size);
    void resizeOutputBuffer(int32_t width, int32_t height);
    void resizeAccumulationButter(int32_t width, int32_t height);

 private:
    std::unique_ptr<CudaOutputBuffer<uchar4>> output_buffer_;
    LaunchParameters launch_parameters_;
    LaunchParameters* d_params_;

    vec3 clear_color_;
};
}
