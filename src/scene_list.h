#pragma once
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "assets/camera.h"
#include "core/glm.h"

namespace assets {
class Model;
class Texture;
}  // namespace assets

typedef std::tuple<std::vector<assets::Model>, std::vector<assets::Texture>> SceneAssets;

class SceneList final {
public:
    static SceneAssets rayTracingInOneWeekend(CameraState& camera);
    static SceneAssets lucyInOneWeekend(CameraState& camera);
    static SceneAssets cornellBox(CameraState& camera);
    static SceneAssets cornellBoxLucy(CameraState& camera);

    static const std::vector<std::pair<std::string, std::function<SceneAssets(CameraState&)>>> allScenes;
};
