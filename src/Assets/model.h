#pragma once

#include "material.h"
#include "procedural.h"
#include "vertex.h"
#include <memory>
#include <string>
#include <vector>

namespace assets {
class Model final {
public:

    static Model loadModel(const std::string& filename);
    static Model createCornellBox(const float scale);
    static Model createBox(const glm::vec3& p0, const glm::vec3& p1, const Material& material);
    static Model createSphere(const glm::vec3& center, float radius, const Material& material, const bool isProcedural);

    Model& operator=(const Model&) = delete;
    Model& operator=(Model&&) = delete;

    Model() = default;
    Model(const Model&) = default;
    Model(Model&&) = default;
    ~Model() = default;

    void setMaterial(const Material& material);
    void transform(const glm::mat4& transform);

    const std::vector<Vertex>& Vertices() const { return vertices_; }
    const std::vector<uint32_t>& Indices() const { return indices_; }
    const std::vector<Material>& Materials() const { return materials_; }

    const class Procedural* Procedural() const { return procedural_.get(); }

    uint32_t NumberOfVertices() const { return static_cast<uint32_t>(vertices_.size()); }
    uint32_t NumberOfIndices() const { return static_cast<uint32_t>(indices_.size()); }
    uint32_t NumberOfMaterials() const { return static_cast<uint32_t>(materials_.size()); }

private:

    Model(std::vector<Vertex>&& vertices,
          std::vector<uint32_t>&& indices,
          std::vector<Material>&& materials,
          const class Procedural* procedural);

    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
    std::vector<Material> materials_;
    std::shared_ptr<const class Procedural> procedural_;
};

}
