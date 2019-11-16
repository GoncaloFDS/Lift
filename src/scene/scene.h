#pragma once
#include "mesh.h"
#include "cuda/material_data.h"
#include <cuda/light.h>
#include "scene/cameras/camera.h"

namespace tinygltf {
class Node;
class Model;
}

constexpr int k_RayTypeCount = 2;
constexpr int k_NumPayloadValues = 4;
constexpr int k_MaxTraceDepth = 4;

namespace lift {
class Camera;

class Scene {
public:

    void addCamera(const std::shared_ptr<Camera>& camera) { cameras_.push_back(camera); }

    void addMesh(const std::shared_ptr<Mesh>& mesh) { meshes_.push_back(mesh); }

    void addMaterial(const MaterialData& material) { materials_.push_back(material); }

    void addBuffer(const uint64_t buf_size, const void* data);
    void addImage(const int32_t width, const int32_t height, const int32_t bits_per_component,
                  const int32_t num_components, const void* data);
    void addSampler(cudaTextureAddressMode address_s, cudaTextureAddressMode address_t,
                    cudaTextureFilterMode filter_mode, const int32_t image_idx);

    void addLight(const Light& light) { lights_.push_back(light); }

    [[nodiscard]] auto buffer(int32_t buffer_index) const -> CUdeviceptr { return buffers_[buffer_index]; }
    [[nodiscard]] auto image(int32_t image_index) const -> cudaArray_t { return images_[image_index]; }
    [[nodiscard]] auto sampler(int32_t sampler_index) const -> cudaTextureObject_t { return samplers_[sampler_index]; }

    void finalize();
    void cleanup();

    [[nodiscard]] auto camera() -> std::shared_ptr<Camera>;
    [[nodiscard]] auto pipeline() const -> OptixPipeline { return pipeline_; }

    [[nodiscard]] auto sbt() const -> const OptixShaderBindingTable* { return &sbt_; }

    [[nodiscard]] auto traversableHandle() const -> OptixTraversableHandle { return ias_handle_; }

    [[nodiscard]] auto aabb() const -> Aabb { return scene_aabb_; }

    [[nodiscard]] auto context() const -> OptixDeviceContext { return context_; }

    [[nodiscard]] auto materials() const -> const std::vector<MaterialData>& { return materials_; }

    [[nodiscard]] auto meshes() const -> const std::vector<std::shared_ptr<Mesh>>& { return meshes_; }

    [[nodiscard]] auto lights() const -> const std::vector<Light>& { return lights_; }

    void createContext();
    void buildMeshAccels();
    void buildInstanceAccel(int ray_type_count = k_RayTypeCount);
    void loadFromFile(const std::string& file_name);
private:
    void createPtxModule();
    void createProgramGroups();
    void createPipeline();
    void createSbt();
    void processGltfNode(const tinygltf::Model& model, const tinygltf::Node& gltf_node, const mat4& parent_matrix);

    std::vector<std::shared_ptr<Camera>> cameras_;
    std::vector<std::shared_ptr<Mesh>> meshes_;
    std::vector<Light> lights_;
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
