#pragma once
#include "renderer_api.h"
#include "cuda_output_buffer.h"
#include <cuda/launch_parameters.h>
#include <platform/opengl/opengl_display.h>
#include <GLFW/glfw3.h>
#include <scene/cameras/camera.h>

constexpr int k_RayTypeCount = 2;

namespace lift {
class Scene;
class Window;

class Renderer {
 public:
    static auto getApi() -> RendererApi::API { return RendererApi::getApi(); }

    void init(CudaOutputBufferType type, ivec2 frame_size);

    void initOptix(const Scene& scene);

    void launchSubframe(const Scene& scene, const ivec2& size);

    void displaySubframe(OpenGLDisplay& gl_display, void* window);

    void onResize(int32_t width, int32_t height);

    void allocLights(Scene& scene);

    void updateLaunchParameters(Scene scene);

	[[nodiscard]] auto pipeline() const -> OptixPipeline { return pipeline_; }
	[[nodiscard]] auto sbt() const -> const OptixShaderBindingTable * { return &sbt_; }
	[[nodiscard]] auto traversableHandle() const -> OptixTraversableHandle { return ias_handle_; }
	[[nodiscard]] auto context() const -> OptixDeviceContext { return context_; }

	void setClearColor(const vec3& color);
    auto clearColor() -> vec3;
    void resetFrame();
 private:
    void createOutputBuffer(CudaOutputBufferType type, ivec2 frame_size);
    void resizeOutputBuffer(int32_t width, int32_t height);
    void resizeAccumulationButter(int32_t width, int32_t height);

	void createContext();
	void buildMeshAccels(const Scene& scene);
	void buildInstanceAccel(const Scene& scene, int ray_type_count = k_RayTypeCount);
	void createPtxModule();
	void createProgramGroups();
	void createPipeline();
	void createSbt(const Scene& scene);

private:
	std::unique_ptr<CudaOutputBuffer<uchar4>> output_buffer_;
    LaunchParameters launch_parameters_;
    LaunchParameters* d_params_;

    vec3 clear_color_;

	OptixDeviceContext context_ = nullptr;
	OptixShaderBindingTable sbt_ = {};
	OptixPipelineCompileOptions pipeline_compile_options_ = {};
	OptixPipeline pipeline_ = nullptr;
	OptixModule ptx_module_ = nullptr;

	OptixProgramGroup raygen_prog_group_ = nullptr;
	OptixProgramGroup radiance_miss_group_ = nullptr;
	OptixProgramGroup occlusion_miss_group_ = nullptr;
	OptixProgramGroup radiance_hit_group_ = nullptr;
	OptixProgramGroup occlusion_hit_group_ = nullptr;
	OptixTraversableHandle ias_handle_ = 0;
	CUdeviceptr d_ias_output_buffer_ = 0;
};
}
