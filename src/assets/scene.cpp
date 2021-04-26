#include "scene.h"
#include "lights.h"
#include "model.h"
#include "sphere.h"
#include "texture.h"
#include "texture_image.h"
#include "vulkan/buffer.h"
#include "vulkan/command_pool.h"
#include "vulkan/image_view.h"
#include <vulkan/buffer_util.hpp>

namespace assets {

Scene::Scene(vulkan::CommandPool& command_pool, SceneAssets& scene_assets)
    : models_(std::move(scene_assets.models)), lights_(std::move(scene_assets.lights)),
      textures_(std::move(scene_assets.textures)) {

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<glm::vec4> procedurals;
    std::vector<VkAabbPositionsKHR> aabbs;
    std::vector<glm::uvec2> offsets;

    for (auto& material : scene_assets.materials) {
        materials_.push_back(material.second);
    }

    for (const auto& model : models_) {
        const auto index_offset = static_cast<uint32_t>(indices.size());
        const auto vertex_offset = static_cast<uint32_t>(vertices.size());

        offsets.emplace_back(index_offset, vertex_offset);

        vertices.insert(vertices.end(), model.vertices().begin(), model.vertices().end());
        indices.insert(indices.end(), model.indices().begin(), model.indices().end());

        for (size_t i = vertex_offset; i != vertices.size(); ++i) {
            auto material_index =
                std::distance(scene_assets.materials.begin(), scene_assets.materials.find(model.materialId()));
            vertices[i].material_index = static_cast<int32_t>(material_index);
        }

        const auto sphere = dynamic_cast<const Sphere*>(model.procedural());
        if (sphere != nullptr) {
            const auto aabb = sphere->boundingBox();
            aabbs.push_back({aabb.first.x, aabb.first.y, aabb.first.z, aabb.second.x, aabb.second.y, aabb.second.z});
            procedurals.emplace_back(sphere->center, sphere->radius);
        } else {
            aabbs.emplace_back();
            procedurals.emplace_back();
        }
    }

    const auto flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    vulkan::BufferUtil::createDeviceBuffer(command_pool,
                       "Vertices",
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | flag,
                       vertices,
                       vertex_buffer_,
                       vertex_buffer_memory_);
    vulkan::BufferUtil::createDeviceBuffer(command_pool,
                       "indices",
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT | flag,
                       indices,
                       index_buffer_,
                       index_buffer_memory_);
    vulkan::BufferUtil::createDeviceBuffer(command_pool, "materials", flag, materials_, material_buffer_, material_buffer_memory_);
    vulkan::BufferUtil::createDeviceBuffer(command_pool, "lights", flag, lights_, light_buffer_, light_buffer_memory_);
    vulkan::BufferUtil::createDeviceBuffer(command_pool, "offsets", flag, offsets, offset_buffer_, offset_buffer_memory_);

    vulkan::BufferUtil::createDeviceBuffer(command_pool, "aabbs", flag, aabbs, aabb_buffer_, aabb_buffer_memory_);
    vulkan::BufferUtil::createDeviceBuffer(command_pool, "procedurals", flag, procedurals, procedural_buffer_, procedural_buffer_memory_);

    // Upload all textures
    texture_images_.reserve(textures_.size());
    texture_image_view_handles_.resize(textures_.size());
    texture_sampler_handles_.resize(textures_.size());

    for (size_t i = 0; i != textures_.size(); ++i) {
        texture_images_.emplace_back(new TextureImage(command_pool, textures_[i]));
        texture_image_view_handles_[i] = texture_images_[i]->imageView().handle();
        texture_sampler_handles_[i] = texture_images_[i]->sampler().handle();
    }
}

Scene::~Scene() {
    texture_sampler_handles_.clear();
    texture_image_view_handles_.clear();
    texture_images_.clear();
    procedural_buffer_.reset();
    procedural_buffer_memory_.reset();
    aabb_buffer_.reset();
    aabb_buffer_memory_.reset();
    offset_buffer_.reset();
    offset_buffer_memory_.reset();
    material_buffer_.reset();
    material_buffer_memory_.reset();
    light_buffer_.reset();
    light_buffer_memory_.reset();
    index_buffer_.reset();
    index_buffer_memory_.reset();
    vertex_buffer_.reset();
    vertex_buffer_memory_.reset();
}

}  // namespace assets
