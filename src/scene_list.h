#pragma once
#include "core/glm.h"
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace assets {
class Model;
class Texture;
}  // namespace assets

typedef std::tuple<std::vector<assets::Model>, std::vector<assets::Texture>> SceneAssets;

class SceneList final {
public:
    struct CameraInitialState {
        glm::mat4 model_view;
        float field_of_view;
        float aperture;
        float focus_distance;
        bool gamma_correction;
        bool has_sky;
    };

    static SceneAssets rayTracingInOneWeekend(CameraInitialState& camera);
    static SceneAssets lucyInOneWeekend(CameraInitialState& camera);
    static SceneAssets cornellBox(CameraInitialState& camera);
    static SceneAssets cornellBoxLucy(CameraInitialState& camera);

    static const std::vector<std::pair<std::string, std::function<SceneAssets(CameraInitialState&)>>> allScenes;
};
