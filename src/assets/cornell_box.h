#pragma once

#include "material.h"
#include "vertex.h"
#include <scene_list.h>
#include <vector>

namespace assets {

class CornellBox final {
public:
    static void create(const float scale, SceneAssets& scene_assets, bool has_light_cover);
};

}  // namespace assets
