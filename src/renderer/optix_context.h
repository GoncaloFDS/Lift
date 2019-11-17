#include <scene/scene.h>
namespace lift {

constexpr int k_RayTypeCount = 2;

class OptixContext {
public:
	void init(const Scene& scene);

	[[nodiscard]] auto pipeline() const -> OptixPipeline { return pipeline_; }
	[[nodiscard]] auto sbt() const -> const OptixShaderBindingTable * { return &sbt_; }
	[[nodiscard]] auto traversableHandle() const -> OptixTraversableHandle { return ias_handle_; }
	[[nodiscard]] auto context() const -> OptixDeviceContext { return context_; }

private:
	void createContext();
	void buildMeshAccels(const Scene& scene);
	void buildInstanceAccel(const Scene& scene, int ray_type_count = k_RayTypeCount);
	void createPtxModule();
	void createProgramGroups();
	void createPipeline();
	void createSbt(const Scene& scene);

private:
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
