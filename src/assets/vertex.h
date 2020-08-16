#pragma once

#include "core/glm.h"
#include "core/utilities.h"
#include <array>

namespace assets {

struct Vertex final {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 tex_coords;
    int32_t material_index;

    bool operator==(const Vertex& other) const {
        return position == other.position && normal == other.normal && tex_coords == other.tex_coords &&
               material_index == other.material_index;
    }

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription binding_description = {};
        binding_description.binding = 0;
        binding_description.stride = sizeof(Vertex);
        binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return binding_description;
    }

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 4> attribute_descriptions = {};

        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].location = 0;
        attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descriptions[0].offset = offsetof(Vertex, position);

        attribute_descriptions[1].binding = 0;
        attribute_descriptions[1].location = 1;
        attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descriptions[1].offset = offsetof(Vertex, normal);

        attribute_descriptions[2].binding = 0;
        attribute_descriptions[2].location = 2;
        attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attribute_descriptions[2].offset = offsetof(Vertex, tex_coords);

        attribute_descriptions[3].binding = 0;
        attribute_descriptions[3].location = 3;
        attribute_descriptions[3].format = VK_FORMAT_R32_SINT;
        attribute_descriptions[3].offset = offsetof(Vertex, material_index);

        return attribute_descriptions;
    }
};

}  // namespace assets

namespace std {
template<>
struct hash<assets::Vertex> final {
    size_t operator()(assets::Vertex const& vertex) const noexcept {
        auto h1 = combine(hash<glm::vec2>()(vertex.tex_coords), hash<int>()(vertex.material_index));
        auto h2 = combine(hash<glm::vec3>()(vertex.normal), h1);
        return combine(hash<glm::vec3>()(vertex.position), h2);
    }

private:
    static size_t combine(size_t hash0, size_t hash1) {
        return hash0 ^ (hash1 + 0x9e3779b9 + (hash0 << 6) + (hash0 >> 2));
    }
};

}  // namespace std
