#pragma once

#include "core/glm.h"
#include "core/utilities.h"
#include <array>

namespace assets {

struct Vertex final {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    int32_t materialIndex;

    bool operator==(const Vertex &other) const {
        return position == other.position && normal == other.normal && texCoord == other.texCoord
            && materialIndex == other.materialIndex;
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
        attribute_descriptions[2].offset = offsetof(Vertex, texCoord);

        attribute_descriptions[3].binding = 0;
        attribute_descriptions[3].location = 3;
        attribute_descriptions[3].format = VK_FORMAT_R32_SINT;
        attribute_descriptions[3].offset = offsetof(Vertex, materialIndex);

        return attribute_descriptions;
    }
};

}  // namespace assets
