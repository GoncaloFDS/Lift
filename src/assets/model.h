#pragma once

#include "material.h"
#include "procedural.h"
#include "vertex.h"
#include <memory>
#include <scene_list.h>
#include <string>
#include <utility>
#include <vector>

namespace assets {
class Model final {
public:
    static Model loadModel(const std::string& filename);
    static Model createBox(const glm::vec3& p0, const glm::vec3& p1, std::string material_id);
    static Model createSphere(const glm::vec3& center, float radius, const std::string material_id, const bool is_procedural);

    Model& operator=(const Model&) = delete;
    Model& operator=(Model&&) = delete;

    Model() = default;
    Model(const Model&) = default;
    Model(Model&&) = default;
    ~Model() = default;

    void setMaterial(std::string material_id) { material_id_ = std::move(material_id); };
    void transform(const glm::mat4& transform);

    [[nodiscard]] const std::string& materialId() const { return material_id_; }
    [[nodiscard]] const std::vector<Vertex>& vertices() const { return vertices_; }
    [[nodiscard]] const std::vector<uint32_t>& indices() const { return indices_; }

    [[nodiscard]] const class Procedural* procedural() const { return procedural_.get(); }

    [[nodiscard]] uint32_t vertexCount() const { return static_cast<uint32_t>(vertices_.size()); }
    [[nodiscard]] uint32_t indexCount() const { return static_cast<uint32_t>(indices_.size()); }

    Model(std::vector<Vertex>&& vertices,
          std::vector<uint32_t>&& indices,
          std::string material_id,
          const class Procedural* procedural);

private:
    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
    std::string material_id_;
    std::shared_ptr<const class Procedural> procedural_;
};

}  // namespace assets
