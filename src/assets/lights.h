#pragma once

#include <core/glm.h>

struct ParallelogramLight final {
    glm::vec4 corner;
    glm::vec4 v1;
    glm::vec4 v2;
    glm::vec4 normal;
    glm::vec4 emission;
};
