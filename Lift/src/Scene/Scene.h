#pragma once
#include "Mesh.h"
#include "MaterialData.h"
#include "scene/cameras/Camera.h"

namespace tinygltf {
	class Node;
	class Model;
}

constexpr int RAY_TYPE_COUNT = 2;
constexpr int NUM_PAYLOAD_VALUES = 4;

namespace lift {
	class Camera;

	class Scene {
	public:

		void AddCamera(const Camera& camera) { cameras_.push_back(camera); }
		void AddMesh(const std::shared_ptr<Mesh>& mesh) { meshes_.push_back(mesh); }
		void AddMaterial(const MaterialData& material) { materials_.push_back(material); }
		void AddBuffer(uint64_t buf_size, const void* data);
		void AddImage(int32_t width, int32_t height, int32_t bits_per_component,
					  int32_t num_components, const void* data);
		void AddSampler(cudaTextureAddressMode address_s, cudaTextureAddressMode address_t,
						cudaTextureFilterMode filter_mode, int32_t image_idx);

		[[nodiscard]] CUdeviceptr GetBuffer(int32_t buffer_index) const { return buffers_[buffer_index]; }
		[[nodiscard]] cudaArray_t GetImage(int32_t image_index) const { return images_[image_index]; }
		[[nodiscard]] cudaTextureObject_t GetSampler(int32_t sampler_index) const { return samplers_[sampler_index]; }

		void Finalize();
		void Cleanup();

		[[nodiscard]] Camera GetCamera() const;
		[[nodiscard]] OptixPipeline GetPipeline() const { return pipeline_; }
		[[nodiscard]] const OptixShaderBindingTable* GetSbt() const { return &sbt_; }
		[[nodiscard]] OptixTraversableHandle GetTraversableHandle() const { return ias_handle_; }
		[[nodiscard]] Aabb GetAabb() const { return scene_aabb_; }
		[[nodiscard]] OptixDeviceContext GetContext() const { return context_; }
		[[nodiscard]] const std::vector<MaterialData>& Materials() const { return materials_; }
		[[nodiscard]] const std::vector<std::shared_ptr<Mesh>>& GetMeshes() const { return meshes_; }

		void CreateContext();
		void BuildMeshAccels();
		void BuildInstanceAccel(int ray_type_count = RAY_TYPE_COUNT);
		void LoadFromFile(const std::string& file_name);
	private:
		void CreatePtxModule();
		void CreateProgramGroups();
		void CreatePipeline();
		void CreateSBT();
		void ProcessGltfNode(const tinygltf::Model& model, const tinygltf::Node& gltf_node, mat4 parent_matrix);

		std::vector<Camera> cameras_;
		std::vector<std::shared_ptr<Mesh>> meshes_;
		std::vector<MaterialData> materials_;
		std::vector<CUdeviceptr> buffers_;
		std::vector<cudaTextureObject_t> samplers_;
		std::vector<cudaArray_t> images_;
		Aabb scene_aabb_;

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
