#pragma once
#include "Utilities/Glm.hpp"
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace assets {
class Model;
class Texture;
}

typedef std::tuple<std::vector<assets::Model>, std::vector<assets::Texture>> SceneAssets;

class SceneList final {
public:

    struct CameraInitialSate {
        glm::mat4 modelView;
        float fieldOfView;
        float aperture;
        float focusDistance;
        bool gammaCorrection;
        bool hasSky;
    };

    static SceneAssets cubeAndSpheres(CameraInitialSate& camera);
    static SceneAssets rayTracingInOneWeekend(CameraInitialSate& camera);
    static SceneAssets planetsInOneWeekend(CameraInitialSate& camera);
    static SceneAssets lucyInOneWeekend(CameraInitialSate& camera);
    static SceneAssets cornellBox(CameraInitialSate& camera);
    static SceneAssets cornellBoxLucy(CameraInitialSate& camera);

    static const std::vector<std::pair<std::string, std::function<SceneAssets(CameraInitialSate&)>>> allScenes;
};
