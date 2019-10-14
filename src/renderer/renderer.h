#pragma once
#include "renderer_api.h"
#include "cuda_output_buffer.h"
#include <cuda/launch_parameters.h>
#include <platform/opengl/opengl_display.h>
#include <GLFW/glfw3.h>

namespace lift {
class Scene;
class Window;

class Renderer {
 public:
    void launchSubframe(const Scene& scene, LaunchParameters& params,
                        LaunchParameters* d_params, const ivec2& size);
    void displaySubframe(OpenGLDisplay& gl_display, void* window);

    void createOutputBuffer(CUDAOutputBufferType type, int32_t width, int32_t height);
    void resizeOutputBuffer(int32_t width, int32_t height);

    static void submit(const std::shared_ptr<VertexArray>& vertex_array);
    static RendererApi::API getApi() { return RendererApi::getApi(); }

 private:
    std::unique_ptr<CudaOutputBuffer<uchar4>> output_buffer_;
};
}
