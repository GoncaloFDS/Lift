#include "model.h"
#include "cornell_box.h"
#include "procedural.h"
#include "sphere.h"
#include <filesystem>
#include <glm/ext.hpp>
#include <glm/gtx/hash.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <core/log.h>
#include <tiny_obj_loader.h>

using namespace glm;

namespace std {
template<>
struct hash<assets::Vertex> final {
  size_t operator()(assets::Vertex const &vertex) const noexcept {
    return combine(hash<vec3>()(vertex.position),
                   combine(hash<vec3>()(vertex.normal),
                           combine(hash<vec2>()(vertex.texCoord),
                                   hash<int>()(vertex.materialIndex))));
  }

  private:
  static size_t combine(size_t hash0, size_t hash1) {
    return hash0 ^ (hash1 + 0x9e3779b9 + (hash0 << 6) + (hash0 >> 2));
  }
};
}// namespace std

namespace assets {

Model Model::loadModel(const std::string &filename) {
  LF_WARN("Loading model {0}", filename);

  const auto timer = std::chrono::high_resolution_clock::now();
  const std::string material_path = std::filesystem::path(filename).parent_path().string();

  tinyobj::attrib_t obj_attrib;
  std::vector<tinyobj::shape_t> obj_shapes;
  std::vector<tinyobj::material_t> obj_materials;
  std::string warn;
  std::string err;

  if (!tinyobj::LoadObj(
        &obj_attrib, &obj_shapes, &obj_materials, &warn, &err,
        filename.c_str(),
        material_path.c_str())) {
    LF_ASSERT(false, "failed to load model");
  }

  if (!warn.empty()) {
    LF_ERROR("Model Load: {0}", warn);
  }

  std::vector<Material> materials;

  for (const auto &material : obj_materials) {
    Material m{};

    m.diffuse = vec4(material.diffuse[0], material.diffuse[1], material.diffuse[2], 1.0);
    m.diffuseTextureId = -1;

    materials.emplace_back(m);
  }

  if (materials.empty()) {
    Material m{};

    m.diffuse = vec4(0.7f, 0.7f, 0.7f, 1.0);
    m.diffuseTextureId = -1;

    materials.emplace_back(m);
  }

  // Geometry
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  std::unordered_map<Vertex, uint32_t> unique_vertices(obj_attrib.vertices.size());
  size_t face_id = 0;

  for (const auto &shape : obj_shapes) {
    const auto &mesh = shape.mesh;

    for (const auto &index : mesh.indices) {
      Vertex vertex = {};

      vertex.position = {
        obj_attrib.vertices[3 * index.vertex_index + 0],
        obj_attrib.vertices[3 * index.vertex_index + 1],
        obj_attrib.vertices[3 * index.vertex_index + 2],
      };

      vertex.normal = {
        obj_attrib.normals[3 * index.normal_index + 0],
        obj_attrib.normals[3 * index.normal_index + 1],
        obj_attrib.normals[3 * index.normal_index + 2]};

      if (!obj_attrib.texcoords.empty()) {
        vertex.texCoord = {
          obj_attrib.texcoords[2 * index.texcoord_index + 0],
          1 - obj_attrib.texcoords[2 * index.texcoord_index + 1]};
      }

      vertex.materialIndex = std::max(0, mesh.material_ids[face_id++ / 3]);

      if (unique_vertices.count(vertex) == 0) {
        unique_vertices[vertex] = static_cast<uint32_t>(vertices.size());
        vertices.push_back(vertex);
      }

      indices.push_back(unique_vertices[vertex]);
    }
  }

  const auto elapsed = std::chrono::duration<float, std::chrono::seconds::period>(
                         std::chrono::high_resolution_clock::now() - timer)
                         .count();

  LF_INFO("Loaded model {0} in {1} -> {2} vertices and {3} materials",
          filename,
          elapsed,
          obj_attrib.vertices.size(),
          materials.size());

  return Model(std::move(vertices), std::move(indices), std::move(materials), nullptr);
}

Model Model::createCornellBox(const float scale) {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  std::vector<Material> materials;

  CornellBox::create(scale, vertices, indices, materials);

  return Model(std::move(vertices),
               std::move(indices),
               std::move(materials),
               nullptr);
}

Model Model::createBox(const vec3 &p0, const vec3 &p1, const Material &material) {
  std::vector<Vertex> vertices = {
    Vertex{vec3(p0.x, p0.y, p0.z), vec3(-1, 0, 0), vec2(0), 0},
    Vertex{vec3(p0.x, p0.y, p1.z), vec3(-1, 0, 0), vec2(0), 0},
    Vertex{vec3(p0.x, p1.y, p1.z), vec3(-1, 0, 0), vec2(0), 0},
    Vertex{vec3(p0.x, p1.y, p0.z), vec3(-1, 0, 0), vec2(0), 0},

    Vertex{vec3(p1.x, p0.y, p1.z), vec3(1, 0, 0), vec2(0), 0},
    Vertex{vec3(p1.x, p0.y, p0.z), vec3(1, 0, 0), vec2(0), 0},
    Vertex{vec3(p1.x, p1.y, p0.z), vec3(1, 0, 0), vec2(0), 0},
    Vertex{vec3(p1.x, p1.y, p1.z), vec3(1, 0, 0), vec2(0), 0},

    Vertex{vec3(p1.x, p0.y, p0.z), vec3(0, 0, -1), vec2(0), 0},
    Vertex{vec3(p0.x, p0.y, p0.z), vec3(0, 0, -1), vec2(0), 0},
    Vertex{vec3(p0.x, p1.y, p0.z), vec3(0, 0, -1), vec2(0), 0},
    Vertex{vec3(p1.x, p1.y, p0.z), vec3(0, 0, -1), vec2(0), 0},

    Vertex{vec3(p0.x, p0.y, p1.z), vec3(0, 0, 1), vec2(0), 0},
    Vertex{vec3(p1.x, p0.y, p1.z), vec3(0, 0, 1), vec2(0), 0},
    Vertex{vec3(p1.x, p1.y, p1.z), vec3(0, 0, 1), vec2(0), 0},
    Vertex{vec3(p0.x, p1.y, p1.z), vec3(0, 0, 1), vec2(0), 0},

    Vertex{vec3(p0.x, p0.y, p0.z), vec3(0, -1, 0), vec2(0), 0},
    Vertex{vec3(p1.x, p0.y, p0.z), vec3(0, -1, 0), vec2(0), 0},
    Vertex{vec3(p1.x, p0.y, p1.z), vec3(0, -1, 0), vec2(0), 0},
    Vertex{vec3(p0.x, p0.y, p1.z), vec3(0, -1, 0), vec2(0), 0},

    Vertex{vec3(p1.x, p1.y, p0.z), vec3(0, 1, 0), vec2(0), 0},
    Vertex{vec3(p0.x, p1.y, p0.z), vec3(0, 1, 0), vec2(0), 0},
    Vertex{vec3(p0.x, p1.y, p1.z), vec3(0, 1, 0), vec2(0), 0},
    Vertex{vec3(p1.x, p1.y, p1.z), vec3(0, 1, 0), vec2(0), 0},
  };

  std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3,
                                   4, 5, 6, 4, 6, 7,
                                   8, 9, 10, 8, 10, 11,
                                   12, 13, 14, 12, 14, 15,
                                   16, 17, 18, 16, 18, 19,
                                   20, 21, 22, 20, 22, 23};

  return Model(std::move(vertices),
               std::move(indices),
               std::vector<Material>{material},
               nullptr);
}

Model Model::createSphere(const vec3 &center, float radius, const Material &material, const bool is_procedural) {
  const int slices = 32;
  const int stacks = 16;

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  const float pi = 3.14159265358979f;

  for (int j = 0; j <= stacks; ++j) {
    const float j0 = pi * j / stacks;

    // Vertex
    const float v = radius * -std::sin(j0);
    const float z = radius * std::cos(j0);

    // Normals
    const float n0 = -std::sin(j0);
    const float n1 = std::cos(j0);

    for (int i = 0; i <= slices; ++i) {
      const float i0 = 2 * pi * i / slices;

      const vec3 position(center.x + v * std::sin(i0),
                          center.y + z,
                          center.z + v * std::cos(i0));

      const vec3 normal(n0 * std::sin(i0),
                        n1,
                        n0 * std::cos(i0));

      const vec2 tex_coord(static_cast<float>(i) / slices,
                           static_cast<float>(j) / stacks);

      vertices.push_back(Vertex{position, normal, tex_coord, 0});
    }
  }

  for (int j = 0; j < stacks; ++j) {
    for (int i = 0; i < slices; ++i) {
      const auto j0 = (j + 0) * (slices + 1);
      const auto j1 = (j + 1) * (slices + 1);
      const auto i0 = i + 0;
      const auto i1 = i + 1;

      indices.push_back(j0 + i0);
      indices.push_back(j1 + i0);
      indices.push_back(j1 + i1);

      indices.push_back(j0 + i0);
      indices.push_back(j1 + i1);
      indices.push_back(j0 + i1);
    }
  }

  return Model(std::move(vertices),
               std::move(indices),
               std::vector<Material>{material},
               is_procedural ? new Sphere(center, radius) : nullptr);
}

void Model::setMaterial(const Material &material) {
  LF_ASSERT(materials_.size() == 1, ("cannot change material on a multi-material model"));

  materials_[0] = material;
}

void Model::transform(const mat4 &transform) {
  const auto inverse_transpose = inverseTranspose(transform);

  for (auto &vertex : vertices_) {
    vertex.position = transform * vec4(vertex.position, 1);
    vertex.normal = inverse_transpose * vec4(vertex.normal, 0);
  }
}

Model::Model(std::vector<Vertex> &&vertices,
             std::vector<uint32_t> &&indices,
             std::vector<Material> &&materials,
             const class Procedural *procedural) : vertices_(std::move(vertices)),
                                                   indices_(std::move(indices)),
                                                   materials_(std::move(materials)),
                                                   procedural_(procedural) {
}

}// namespace assets
