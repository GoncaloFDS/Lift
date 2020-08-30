#include "model.h"
#include "cornell_box.h"
#include "sphere.h"
#include <filesystem>
#include <utility>

#define TINYOBJLOADER_IMPLEMENTATION
#include <core/log.h>
#include <core/profiler.h>
#include <pbrtParser/Scene.h>
#include <tiny_obj_loader.h>

using namespace glm;

namespace assets {

Model Model::loadModel(const std::string& filename) {
    LF_INFO("Loading model {0}", filename);
    Profiler profiler("Loading Model took");

    const std::string material_path = std::filesystem::path(filename).parent_path().string();

    tinyobj::attrib_t obj_attrib;
    std::vector<tinyobj::shape_t> obj_shapes;
    std::vector<tinyobj::material_t> obj_materials;
    std::string warn;
    std::string err;

    if (!tinyobj::LoadObj(&obj_attrib,
                          &obj_shapes,
                          &obj_materials,
                          &warn,
                          &err,
                          filename.c_str(),
                          material_path.c_str())) {
        LF_FATAL("failed to load model: {}", filename);
        LF_ASSERT(false, "failed to load model");
    }

    if (!warn.empty()) {
        LF_ERROR("Warning: {0}", warn);
    }

    if (!err.empty()) {
        LF_FATAL("Error loading obj: {0}", err);
    }

    std::vector<uint32_t> materials;

    // Geometry
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::unordered_map<Vertex, uint32_t> unique_vertices(obj_attrib.vertices.size());

    for (const auto& shape : obj_shapes) {
        const auto& mesh = shape.mesh;

        for (const auto& index : mesh.indices) {
            Vertex vertex = {};

            vertex.position = {
                obj_attrib.vertices[3 * index.vertex_index + 0],
                obj_attrib.vertices[3 * index.vertex_index + 1],
                obj_attrib.vertices[3 * index.vertex_index + 2],
            };

            vertex.normal = {obj_attrib.normals[3 * index.normal_index + 0],
                             obj_attrib.normals[3 * index.normal_index + 1],
                             obj_attrib.normals[3 * index.normal_index + 2]};

            if (!obj_attrib.texcoords.empty()) {
                vertex.tex_coords = {obj_attrib.texcoords[2 * index.texcoord_index + 0],
                                     1 - obj_attrib.texcoords[2 * index.texcoord_index + 1]};
            }

            if (unique_vertices.count(vertex) == 0) {
                unique_vertices[vertex] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
            }

            indices.push_back(unique_vertices[vertex]);
        }
    }
    LF_INFO("\tNumber of triangles: {}", indices.size() / 3);

    return Model(std::move(vertices), std::move(indices), "default", nullptr);
}

Model Model::createBox(const vec3& p0, const vec3& p1, std::string material_id) {
    std::vector<Vertex> vertices = {
        Vertex {vec3(p0.x, p0.y, p0.z), vec3(-1, 0, 0), vec2(0), 0},
        Vertex {vec3(p0.x, p0.y, p1.z), vec3(-1, 0, 0), vec2(0), 0},
        Vertex {vec3(p0.x, p1.y, p1.z), vec3(-1, 0, 0), vec2(0), 0},
        Vertex {vec3(p0.x, p1.y, p0.z), vec3(-1, 0, 0), vec2(0), 0},

        Vertex {vec3(p1.x, p0.y, p1.z), vec3(1, 0, 0), vec2(0), 0},
        Vertex {vec3(p1.x, p0.y, p0.z), vec3(1, 0, 0), vec2(0), 0},
        Vertex {vec3(p1.x, p1.y, p0.z), vec3(1, 0, 0), vec2(0), 0},
        Vertex {vec3(p1.x, p1.y, p1.z), vec3(1, 0, 0), vec2(0), 0},

        Vertex {vec3(p1.x, p0.y, p0.z), vec3(0, 0, -1), vec2(0), 0},
        Vertex {vec3(p0.x, p0.y, p0.z), vec3(0, 0, -1), vec2(0), 0},
        Vertex {vec3(p0.x, p1.y, p0.z), vec3(0, 0, -1), vec2(0), 0},
        Vertex {vec3(p1.x, p1.y, p0.z), vec3(0, 0, -1), vec2(0), 0},

        Vertex {vec3(p0.x, p0.y, p1.z), vec3(0, 0, 1), vec2(0), 0},
        Vertex {vec3(p1.x, p0.y, p1.z), vec3(0, 0, 1), vec2(0), 0},
        Vertex {vec3(p1.x, p1.y, p1.z), vec3(0, 0, 1), vec2(0), 0},
        Vertex {vec3(p0.x, p1.y, p1.z), vec3(0, 0, 1), vec2(0), 0},

        Vertex {vec3(p0.x, p0.y, p0.z), vec3(0, -1, 0), vec2(0), 0},
        Vertex {vec3(p1.x, p0.y, p0.z), vec3(0, -1, 0), vec2(0), 0},
        Vertex {vec3(p1.x, p0.y, p1.z), vec3(0, -1, 0), vec2(0), 0},
        Vertex {vec3(p0.x, p0.y, p1.z), vec3(0, -1, 0), vec2(0), 0},

        Vertex {vec3(p1.x, p1.y, p0.z), vec3(0, 1, 0), vec2(0), 0},
        Vertex {vec3(p0.x, p1.y, p0.z), vec3(0, 1, 0), vec2(0), 0},
        Vertex {vec3(p0.x, p1.y, p1.z), vec3(0, 1, 0), vec2(0), 0},
        Vertex {vec3(p1.x, p1.y, p1.z), vec3(0, 1, 0), vec2(0), 0},
    };

    std::vector<uint32_t> indices = {0,  1,  2,  0,  2,  3,  4,  5,  6,  4,  6,  7,  8,  9,  10, 8,  10, 11,
                                     12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23};

    return Model(std::move(vertices), std::move(indices), std::move(material_id), nullptr);
}

Model Model::createSphere(const vec3& center, float radius, std::string material_id, const bool is_procedural) {
    const int slices = 32;
    const int stacks = 16;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    for (int j = 0; j <= stacks; ++j) {
        const float j0 = glm::pi<float>() * j / stacks;

        // Vertex
        const float v = radius * -std::sin(j0);
        const float z = radius * std::cos(j0);

        // Normals
        const float n0 = -std::sin(j0);
        const float n1 = std::cos(j0);

        for (int i = 0; i <= slices; ++i) {
            const float i0 = 2 * glm::pi<float>() * i / slices;

            const vec3 position(center.x + v * std::sin(i0), center.y + z, center.z + v * std::cos(i0));

            const vec3 normal(n0 * std::sin(i0), n1, n0 * std::cos(i0));

            const vec2 tex_coord(static_cast<float>(i) / slices, static_cast<float>(j) / stacks);

            vertices.push_back(Vertex {position, normal, tex_coord, 0});
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
                 std::move(material_id),
                 is_procedural ? new Sphere(center, radius) : nullptr);
}

void Model::transform(const mat4& transform) {
    const auto inverse_transpose = inverseTranspose(transform);

    for (auto& vertex : vertices_) {
        vertex.position = transform * vec4(vertex.position, 1);
        vertex.normal = inverse_transpose * vec4(vertex.normal, 0);
    }
}

Model::Model(std::vector<Vertex>&& vertices,
             std::vector<uint32_t>&& indices,
             std::string material_id,
             const class Procedural* procedural)
    : vertices_(std::move(vertices)), indices_(std::move(indices)), material_id_(std::move(material_id)),
      procedural_(procedural) {
}

}  // namespace assets
