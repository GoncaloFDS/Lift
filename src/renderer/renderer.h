#pragma once
#include "renderer_api.h"
#include "cuda_buffer.h"
#include <cuda/launch_parameters.h>

namespace lift {
class Scene;

class Renderer {
public:
    Renderer();

    void launchSubframe(const Scene& scene, LaunchParameters& params, const ivec2& size);
    static void submit(const std::shared_ptr<VertexArray>& vertex_array);
    static RendererApi::API getApi() { return RendererApi::getApi(); }

private:

    CudaBuffer<LaunchParameters> d_params_;
};
}
