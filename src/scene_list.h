#pragma once
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "assets/camera.h"
#include "assets/lights.h"
#include "core/glm.h"
#include "pbrtParser/Scene.h"

namespace assets {
class Model;
class Texture;
struct Material;
}  // namespace assets

struct SceneAssets {
    std::vector<assets::Model> models;
    std::map<std::string, assets::Material> materials;
    std::vector<Light> lights;
    std::vector<assets::Texture> textures;
    CameraState camera;
};

class SceneList {
public:
    static SceneAssets cornellBox();
    static SceneAssets teapot();
    static SceneAssets cornellBoxDragon();
    static SceneAssets diningRoom();
    static SceneAssets classroom();
    static SceneAssets bathroom();

    static const std::vector<std::pair<std::string, std::function<SceneAssets()>>> all_scenes;
};

void traverse(const pbrt::Object::SP& object, SceneAssets& scene_assets);
