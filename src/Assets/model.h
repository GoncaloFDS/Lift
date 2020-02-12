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
  static Model loadModel(const std::string &filename);
  static Model createCornellBox(float scale);
  static Model createBox(const glm::vec3 &p0, const glm::vec3 &p1, const Material &material);
  static Model createSphere(const glm::vec3 &center, float radius, const Material &material, bool is_procedural);

  Model &operator=(const Model &) = delete;
  Model &operator=(Model &&) = delete;

  Model() = default;
  Model(const Model &) = default;
  Model(Model &&) = default;
  ~Model() = default;

  void setMaterial(const Material &material);
  void transform(const glm::mat4 &transform);

  [[nodiscard]] const std::vector<Vertex> &vertices() const { return vertices_; }
  [[nodiscard]] const std::vector<uint32_t> &indices() const { return indices_; }
  [[nodiscard]] const std::vector<Material> &materials() const { return materials_; }

  [[nodiscard]] const class Procedural *procedural() const { return procedural_.get(); }

  [[nodiscard]] uint32_t vertexCount() const { return static_cast<uint32_t>(vertices_.size()); }
  [[nodiscard]] uint32_t indexCount() const { return static_cast<uint32_t>(indices_.size()); }
  [[nodiscard]] uint32_t materialCount() const { return static_cast<uint32_t>(materials_.size()); }

  private:
  Model(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices, std::vector<Material> &&materials,
        const class Procedural *procedural);

  std::vector<Vertex> vertices_;
  std::vector<uint32_t> indices_;
  std::vector<Material> materials_;
  std::shared_ptr<const class Procedural> procedural_;
};

}  // namespace assets
