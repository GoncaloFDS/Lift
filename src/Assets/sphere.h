#pragma once

#include "procedural.h"
#include "core/glm.h"

namespace assets {

class Sphere final : public Procedural {
public:

    Sphere(const glm::vec3& center, const float radius) :
        center(center), radius(radius) {
    }

    const glm::vec3 center;
    const float radius;

    [[nodiscard]] std::pair<glm::vec3, glm::vec3> boundingBox() const override {
        return std::make_pair(center - radius, center + radius);
    }

};

}