#pragma once

#include "core/glm.h"
#include <utility>

namespace assets {

class Procedural {
public:
    Procedural(const Procedural&) = delete;
    Procedural(Procedural&&) = delete;
    Procedural& operator=(const Procedural&) = delete;
    Procedural& operator=(Procedural&&) = delete;

    Procedural() = default;
    virtual ~Procedural() = default;
    ;
    virtual std::pair<glm::vec3, glm::vec3> boundingBox() const = 0;
};
}  // namespace assets
