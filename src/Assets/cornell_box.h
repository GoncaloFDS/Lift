#pragma once

#include "material.h"
#include "vertex.h"
#include <vector>

namespace assets {

class CornellBox final {
  public:
  static void create(const float scale, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices,
                     std::vector<Material> &materials);
};

}  // namespace assets
