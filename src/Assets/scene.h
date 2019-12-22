#pragma once

#include "core/utilities.h"
#include <memory>
#include <vector>

namespace vulkan {
class Buffer;
class CommandPool;
class DeviceMemory;
class Image;
}

namespace assets {
class Model;
class Texture;
class TextureImage;

class Scene final {
public:

    Scene(const Scene&) = delete;
    Scene(Scene&&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene& operator=(Scene&&) = delete;

    Scene(vulkan::CommandPool& commandPool,
          std::vector<Model>&& models,
          std::vector<Texture>&& textures,
          bool usedForRayTracing);
    ~Scene();

    const std::vector<Model>& Models() const { return models_; }
    bool HasProcedurals() const { return static_cast<bool>(proceduralBuffer_); }

    const vulkan::Buffer& VertexBuffer() const { return *vertexBuffer_; }
    const vulkan::Buffer& IndexBuffer() const { return *indexBuffer_; }
    const vulkan::Buffer& MaterialBuffer() const { return *materialBuffer_; }
    const vulkan::Buffer& OffsetsBuffer() const { return *offsetBuffer_; }
    const vulkan::Buffer& AabbBuffer() const { return *aabbBuffer_; }
    const vulkan::Buffer& ProceduralBuffer() const { return *proceduralBuffer_; }
    const std::vector<VkImageView> TextureImageViews() const { return textureImageViewHandles_; }
    const std::vector<VkSampler> TextureSamplers() const { return textureSamplerHandles_; }

private:

    const std::vector<Model> models_;
    const std::vector<Texture> textures_;

    std::unique_ptr<vulkan::Buffer> vertexBuffer_;
    std::unique_ptr<vulkan::DeviceMemory> vertexBufferMemory_;

    std::unique_ptr<vulkan::Buffer> indexBuffer_;
    std::unique_ptr<vulkan::DeviceMemory> indexBufferMemory_;

    std::unique_ptr<vulkan::Buffer> materialBuffer_;
    std::unique_ptr<vulkan::DeviceMemory> materialBufferMemory_;

    std::unique_ptr<vulkan::Buffer> offsetBuffer_;
    std::unique_ptr<vulkan::DeviceMemory> offsetBufferMemory_;

    std::unique_ptr<vulkan::Buffer> aabbBuffer_;
    std::unique_ptr<vulkan::DeviceMemory> aabbBufferMemory_;

    std::unique_ptr<vulkan::Buffer> proceduralBuffer_;
    std::unique_ptr<vulkan::DeviceMemory> proceduralBufferMemory_;

    std::vector<std::unique_ptr<TextureImage>> textureImages_;
    std::vector<VkImageView> textureImageViewHandles_;
    std::vector<VkSampler> textureSamplerHandles_;
};

}
