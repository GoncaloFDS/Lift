#pragma once
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "assets/camera.h"
#include "assets/lights.h"
#include "core/glm.h"

namespace assets {
class Model;
class Texture;
}  // namespace assets

struct SceneAssets {
    std::vector<assets::Model> models;
    std::vector<assets::Texture> textures;
    CameraState camera;
    Light light;
};

class SceneList final {
public:
    static SceneAssets rayTracingInOneWeekend();
    static SceneAssets lucyInOneWeekend();
    static SceneAssets teapot();
    static SceneAssets cornellBox();
    static SceneAssets cornellBoxDragon();
    static SceneAssets sponza();
    static SceneAssets pbrt();

    static const std::vector<std::pair<std::string, std::function<SceneAssets()>>> all_scenes;
};
