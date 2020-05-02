#include "scene_list.h"
#include "assets/material.h"
#include "assets/model.h"
#include "assets/texture.h"
#include <random>

using namespace glm;
using assets::Material;
using assets::Model;
using assets::Texture;

const std::vector<std::pair<std::string, std::function<SceneAssets(CameraState&)>>>
    SceneList::allScenes = {
        {"Cornell Box", cornellBox},
        {"Cornell Box & Lucy", cornellBoxLucy},
        {"Ray Tracing In One Weekend", rayTracingInOneWeekend},
        {"Lucy In One Weekend", lucyInOneWeekend},
};

SceneAssets SceneList::rayTracingInOneWeekend(CameraState& camera) {
    camera.eye = vec3(13, 2, 3);
    camera.look_at = vec3(0);
    camera.up = vec3(0, 1, 0);
    camera.field_of_view = 40;
    camera.field_of_view = 20;
    camera.aperture = 0.1f;
    camera.focus_distance = 10.0f;
    camera.gamma_correction = true;
    camera.has_sky = true;

    const bool is_procedural = true;

    std::mt19937 engine(42);
    auto random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<Model> models;

    models.push_back(
        Model::createSphere(vec3(0, -1000, 0), 1000, Material::lambertian(vec3(0.5f, 0.5f, 0.5f)), is_procedural));

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            const float choose_mat = random();
            const vec3 center(a + 0.9f * random(), 0.2f, b + 0.9f * random());

            if (length(center - vec3(4, 0.2f, 0)) > 0.9) {
                if (choose_mat < 0.8f)  // Diffuse
                {
                    models.push_back(Model::createSphere(
                        center,
                        0.2f,
                        Material::lambertian(vec3(random() * random(), random() * random(), random() * random())),
                        is_procedural));
                } else if (choose_mat < 0.95f)  // Metal
                {
                    models.push_back(Model::createSphere(
                        center,
                        0.2f,
                        Material::metallic(vec3(0.5f * (1 + random()), 0.5f * (1 + random()), 0.5f * (1 + random())),
                                           0.5f * random()),
                        is_procedural));
                } else  // Glass
                {
                    models.push_back(Model::createSphere(center, 0.2f, Material::dielectric(1.5f), is_procedural));
                }
            }
        }
    }

    models.push_back(Model::createSphere(vec3(0, 1, 0), 1.0f, Material::dielectric(1.5f), is_procedural));
    models.push_back(
        Model::createSphere(vec3(-4, 1, 0), 1.0f, Material::lambertian(vec3(0.4f, 0.2f, 0.1f)), is_procedural));
    models.push_back(
        Model::createSphere(vec3(4, 1, 0), 1.0f, Material::metallic(vec3(0.7f, 0.6f, 0.5f), 0.0f), is_procedural));

    return std::forward_as_tuple(std::move(models), std::vector<Texture>());
}

SceneAssets SceneList::lucyInOneWeekend(CameraState& camera) {
    camera.eye = vec3(13, 2, 3);
    camera.look_at = vec3(0);
    camera.up = vec3(0, 1, 0);
    camera.field_of_view = 20;
    camera.aperture = 0.05f;
    camera.focus_distance = 10.0f;
    camera.gamma_correction = true;
    camera.has_sky = true;

    const bool is_procedural = true;

    std::mt19937 engine(42);
    auto random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<Model> models;

    models.push_back(
        Model::createSphere(vec3(0, -1000, 0), 1000, Material::lambertian(vec3(0.5f, 0.5f, 0.5f)), is_procedural));

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            const float choose_mat = random();
            const vec3 center(a + 0.9f * random(), 0.2f, b + 0.9f * random());

            if (length(center - vec3(4, 0.2f, 0)) > 0.9) {
                if (choose_mat < 0.8f)  // Diffuse
                {
                    models.push_back(Model::createSphere(
                        center,
                        0.2f,
                        Material::lambertian(vec3(random() * random(), random() * random(), random() * random())),
                        is_procedural));
                } else if (choose_mat < 0.95f)  // Metal
                {
                    models.push_back(Model::createSphere(
                        center,
                        0.2f,
                        Material::metallic(vec3(0.5f * (1 + random()), 0.5f * (1 + random()), 0.5f * (1 + random())),
                                           0.5f * random()),
                        is_procedural));
                } else  // Glass
                {
                    models.push_back(Model::createSphere(center, 0.2f, Material::dielectric(1.5f), is_procedural));
                }
            }
        }
    }

    auto lucy_0 = Model::loadModel("../resources/models/lucy.obj");
    auto lucy_1 = lucy_0;
    auto lucy_2 = lucy_0;

    const auto i = mat4(1);
    const float scale_factor = 0.0035f;

    lucy_0.transform(
        rotate(scale(translate(i, vec3(0, -0.08f, 0)), vec3(scale_factor)), radians(90.0f), vec3(0, 1, 0)));

    lucy_1.transform(
        rotate(scale(translate(i, vec3(-4, -0.08f, 0)), vec3(scale_factor)), radians(90.0f), vec3(0, 1, 0)));

    lucy_2.transform(
        rotate(scale(translate(i, vec3(4, -0.08f, 0)), vec3(scale_factor)), radians(90.0f), vec3(0, 1, 0)));

    lucy_0.setMaterial(Material::dielectric(1.5f));
    lucy_1.setMaterial(Material::lambertian(vec3(0.4f, 0.2f, 0.1f)));
    lucy_2.setMaterial(Material::metallic(vec3(0.7f, 0.6f, 0.5f), 0.05f));

    models.push_back(std::move(lucy_0));
    models.push_back(std::move(lucy_1));
    models.push_back(std::move(lucy_2));

    return std::forward_as_tuple(std::move(models), std::vector<Texture>());
}

SceneAssets SceneList::cornellBox(CameraState& camera) {
    camera.eye = vec3(278, 278, 800);
    camera.look_at = vec3(278, 278, 0);
    camera.up = vec3(0, 1, 0);
    camera.field_of_view = 40;
    camera.aperture = 0.0f;
    camera.focus_distance = 10.0f;
    camera.gamma_correction = true;
    camera.has_sky = false;

    const auto i = mat4(1);
    const auto white = Material::lambertian(vec3(0.73f, 0.73f, 0.73f));
    const auto metal = Material::metallic(vec3(0.7f, 0.6f, 0.5f), 0.05f);
    const auto glass = Material::dielectric(1.5f);

    auto box_0 = Model::createBox(vec3(0, 1, -165), vec3(165, 165, 0), metal);
    auto box_1 = Model::createBox(vec3(0, 1, -165), vec3(165, 330, 0), glass);

    box_0.transform(rotate(translate(i, vec3(555 - 130 - 165, 0, -65)), radians(-18.0f), vec3(0, 1, 0)));
    box_1.transform(rotate(translate(i, vec3(555 - 265 - 165, 0, -295)), radians(15.0f), vec3(0, 1, 0)));

    std::vector<Model> models;
    models.push_back(Model::createCornellBox(555));
    models.push_back(box_0);
    models.push_back(box_1);

    return std::make_tuple(std::move(models), std::vector<Texture>());
}

SceneAssets SceneList::cornellBoxLucy(CameraState& camera) {
    camera.eye = vec3(278, 278, 800);
    camera.look_at = vec3(278, 278, 0);
    camera.up = vec3(0, 1, 0);
    camera.field_of_view = 40;
    camera.aperture = 0.0f;
    camera.focus_distance = 10.0f;
    camera.gamma_correction = true;
    camera.has_sky = false;

    const auto sphere =
        Model::createSphere(vec3(555 - 130, 165.0f, -165.0f / 2 - 65), 80.0f, Material::dielectric(1.5f), true);
    auto lucy_0 = Model::loadModel("../resources/models/lucy.obj");

    lucy_0.transform(rotate(scale(translate(mat4(1), vec3(555 - 300 - 165 / 2, -8, -295 - 165 / 2)), vec3(0.6f)),
                            radians(75.0f),
                            vec3(0, 1, 0)));
    lucy_0.setMaterial(Material::dielectric(1.5f));

    std::vector<Model> models;
    models.push_back(Model::createCornellBox(555));
    models.push_back(sphere);
    models.push_back(lucy_0);

    return std::forward_as_tuple(std::move(models), std::vector<Texture>());
}
