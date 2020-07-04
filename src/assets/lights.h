#pragma once

#include <core/glm.h>

struct Light {
    glm::vec4 corner;
    glm::vec4 v1;
    glm::vec4 v2;
    glm::vec4 normal;
    glm::vec4 emission;
};

struct PathNode {
    glm::vec4 color;
    glm::vec4 position;
    glm::vec4 normal;
};
