#include "pch.h"
#include "renderer.h"
#include "render_command.h"

#include <optix.h>
#include <optix_stubs.h>
#include <core/profiler.h>
#include <platform/opengl/opengl_display.h>
#include <GLFW/glfw3.h>
#include <core/os/window.h>
#include "scene/scene.h"
#include "cuda/launch_parameters.h"

void lift::Renderer::launchSubframe(const Scene& scene, LaunchParameters& params, LaunchParameters* d_params,
                                    const ivec2& size) {
    Profiler profiler(Profiler::Id::Render);
    uchar4* result_buffer_data = output_buffer_->map();
    params.frame_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
                               &params,
                               sizeof(LaunchParameters),
                               cudaMemcpyHostToDevice,
                               nullptr));

    OPTIX_CHECK(optixLaunch(
        scene.pipeline(),
        nullptr,
        reinterpret_cast<CUdeviceptr>( d_params ),
        sizeof(LaunchParameters),
        scene.sbt(),
        size.x,
        size.y,
        1));

    output_buffer_->unmap();
    CUDA_SYNC_CHECK();
}

void lift::Renderer::displaySubframe(OpenGLDisplay& gl_display, void* window) {
    Profiler profiler(Profiler::Id::Display);
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize(static_cast<GLFWwindow*>(window), &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        ivec2(output_buffer_->width(), output_buffer_->height()),
        ivec2(framebuf_res_x, framebuf_res_y),
        output_buffer_->getPBO()
    );
}

void lift::Renderer::submit(const std::shared_ptr<VertexArray>& vertex_array) {
    vertex_array->bind();
    RenderCommand::drawIndexed(vertex_array);

}
void lift::Renderer::createOutputBuffer(CUDAOutputBufferType type, int32_t width, int32_t height) {
    output_buffer_ = std::make_unique<CudaOutputBuffer<uchar4>>(type, width, height);

}
void lift::Renderer::resizeOutputBuffer(int32_t width, int32_t height) {
    output_buffer_->resize(width, height);
}

void lift::Renderer::allocLights(lift::Scene& scene, lift::LaunchParameters& params) {
    auto& lights = scene.lights();
    params.lights.count = lights.size();
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.lights.data),
        lights.size() * sizeof(Lights::PointLight)

    ));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>( params.lights.data ),
        lights.data(),
        lights.size() * sizeof(Lights::PointLight),
        cudaMemcpyHostToDevice
    ));
}
