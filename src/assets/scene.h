#pragma once

#include "core/utilities.h"
#include "lights.h"
#include <memory>
#include <scene_list.h>
#include <vector>

namespace vulkan {
class Buffer;
class CommandPool;
class DeviceMemory;
class Image;
}  // namespace vulkan

namespace assets {
class Model;
struct Material;
class Texture;
class TextureImage;

class Scene final {
public:
    Scene(const Scene&) = delete;
    Scene(Scene&&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene& operator=(Scene&&) = delete;

    Scene(vulkan::CommandPool& command_pool, SceneAssets& scene_assets);
    ~Scene();

    [[nodiscard]] const std::vector<Model>& models() const { return models_; }
    [[nodiscard]] bool hasProcedurals() const { return static_cast<bool>(procedural_buffer_); }

    [[nodiscard]] const vulkan::Buffer& vertexBuffer() const { return *vertex_buffer_; }
    [[nodiscard]] const vulkan::Buffer& indexBuffer() const { return *index_buffer_; }
    [[nodiscard]] const vulkan::Buffer& materialBuffer() const { return *material_buffer_; }
    [[nodiscard]] const vulkan::Buffer& lightBuffer() const { return *light_buffer_; }
    [[nodiscard]] const vulkan::Buffer& offsetsBuffer() const { return *offset_buffer_; }
    [[nodiscard]] const vulkan::Buffer& aabbBuffer() const { return *aabb_buffer_; }
    [[nodiscard]] const vulkan::Buffer& proceduralBuffer() const { return *procedural_buffer_; }
    [[nodiscard]] std::vector<VkImageView> textureImageViews() const { return texture_image_view_handles_; }
    [[nodiscard]] std::vector<VkSampler> textureSamplers() const { return texture_sampler_handles_; }

private:
    std::vector<Model> models_;
    std::vector<Material> materials_;
    std::vector<Light> lights_;
    std::vector<Texture> textures_;

    std::unique_ptr<vulkan::Buffer> vertex_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> vertex_buffer_memory_;

    std::unique_ptr<vulkan::Buffer> index_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> index_buffer_memory_;

    std::unique_ptr<vulkan::Buffer> material_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> material_buffer_memory_;

    std::unique_ptr<vulkan::Buffer> light_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> light_buffer_memory_;

    std::unique_ptr<vulkan::Buffer> offset_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> offset_buffer_memory_;

    std::unique_ptr<vulkan::Buffer> aabb_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> aabb_buffer_memory_;

    std::unique_ptr<vulkan::Buffer> procedural_buffer_;
    std::unique_ptr<vulkan::DeviceMemory> procedural_buffer_memory_;

    std::vector<std::unique_ptr<TextureImage>> texture_images_;
    std::vector<VkImageView> texture_image_view_handles_;
    std::vector<VkSampler> texture_sampler_handles_;
};

}  // namespace assets
