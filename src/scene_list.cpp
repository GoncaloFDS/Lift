#include "scene_list.h"
#include "assets/material.h"
#include "assets/model.h"
#include "assets/texture.h"
#include <effolkronium/random.hpp>

using namespace glm;
using Random = effolkronium::random_static;
using assets::Material;
using assets::Model;
using assets::Texture;

const std::vector<std::pair<std::string, std::function<SceneAssets()>>> SceneList::all_scenes = {
    {"teapot", teapot},
    {"cornell box", cornellBox},
    {"cornell box dragon", cornellBoxDragon},
    {"ray tracing in one weekend", rayTracingInOneWeekend},
    {"lucy in one weekend", lucyInOneWeekend},
};

SceneAssets SceneList::rayTracingInOneWeekend() {
    CameraState camera;
    camera.eye = vec3(13, 2, 3);
    camera.look_at = vec3(0);
    camera.up = vec3(0, 1, 0);
    camera.field_of_view = 40;
    camera.field_of_view = 20;
    camera.aperture = 0.1f;
    camera.focus_distance = 10.0f;
    camera.gamma_correction = true;
    camera.has_sky = true;

    Light light = {
        {213.0f, 553.0f, -328.0f, 0},
        {130.0f, 0.0f, 0.0f, 0},
        {0.0f, 0.0f, 130.0f, 0},
        {0.0f, -1.0f, 0.0f, 0},
        {10.0f, 10.0f, 10.0f, 0},
    };

    const bool is_procedural = true;

    std::vector<Model> models;

    models.push_back(
        Model::createSphere(vec3(0, -1000, 0), 1000, Material::lambertian(vec3(0.5f, 0.5f, 0.5f)), is_procedural));

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            const float choose_mat = Random::get(0.0f, 1.0f);
            const vec3 center(a + 0.9f * Random::get(0.0f, 1.0f), 0.2f, b + 0.9f * Random::get(0.0f, 1.0f));

            if (length(center - vec3(4, 0.2f, 0)) > 0.9) {
                if (choose_mat < 0.8f)  // Diffuse
                {
                    models.push_back(Model::createSphere(
                        center,
                        0.2f,
                        Material::lambertian(vec3(Random::get(0.0f, 1.0f) * Random::get(0.0f, 1.0f),
                                                  Random::get(0.0f, 1.0f) * Random::get(0.0f, 1.0f),
                                                  Random::get(0.0f, 1.0f) * Random::get(0.0f, 1.0f))),
                        is_procedural));
                } else if (choose_mat < 0.95f)  // Metal
                {
                    models.push_back(Model::createSphere(center,
                                                         0.2f,
                                                         Material::metallic(vec3(0.5f * (1 + Random::get(0.0f, 1.0f)),
                                                                                 0.5f * (1 + Random::get(0.0f, 1.0f)),
                                                                                 0.5f * (1 + Random::get(0.0f, 1.0f))),
                                                                            0.5f * Random::get(0.0f, 1.0f)),
                                                         is_procedural));
                } else  // Glass
                {
                    models.push_back(
                        Model::createSphere(center, 0.2f, Material::dielectric(1.5f), is_procedural));
                }
            }
        }
    }

    models.push_back(Model::createSphere(vec3(0, 1, 0), 1.0f, Material::dielectric(1.5f), is_procedural));
    models.push_back(
        Model::createSphere(vec3(-4, 1, 0), 1.0f, Material::lambertian(vec3(0.4f, 0.2f, 0.1f)), is_procedural));
    models.push_back(
        Model::createSphere(vec3(4, 1, 0), 1.0f, Material::metallic(vec3(0.7f, 0.6f, 0.5f), 0.0f), is_procedural));

    return SceneAssets{ std::move(models), std::vector<Texture>(), camera, light};
}

SceneAssets SceneList::lucyInOneWeekend() {
    CameraState camera;
    camera.eye = vec3(13, 2, 3);
    camera.look_at = vec3(0);
    camera.up = vec3(0, 1, 0);
    camera.field_of_view = 20;
    camera.aperture = 0.05f;
    camera.focus_distance = 10.0f;
    camera.gamma_correction = true;
    camera.has_sky = true;

    Light light = {
        {213.0f, 553.0f, -328.0f, 0},
        {130.0f, 0.0f, 0.0f, 0},
        {0.0f, 0.0f, 130.0f, 0},
        {0.0f, -1.0f, 0.0f, 0},
        {10.0f, 10.0f, 10.0f, 0},
    };

    const bool is_procedural = true;

    std::vector<Model> models;

    models.push_back(
        Model::createSphere(vec3(0, -1000, 0), 1000, Material::lambertian(vec3(0.5f, 0.5f, 0.5f)), is_procedural));

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            const float choose_mat = Random::get(0.f, 1.f);
            const vec3 center(a + 0.9f * Random::get(0.f, 1.f), 0.2f, b + 0.9f * Random::get(0.f, 1.f));

            if (length(center - vec3(4, 0.2f, 0)) > 0.9) {
                if (choose_mat < 0.8f)  // Diffuse
                {
                    models.push_back(
                        Model::createSphere(center,
                                            0.2f,
                                            Material::lambertian(vec3(Random::get(0.f, 1.f) * Random::get(0.f, 1.f),
                                                                      Random::get(0.f, 1.f) * Random::get(0.f, 1.f),
                                                                      Random::get(0.f, 1.f) * Random::get(0.f, 1.f))),
                                            is_procedural));
                } else if (choose_mat < 0.95f)  // Metal
                {
                    models.push_back(Model::createSphere(center,
                                                         0.2f,
                                                         Material::metallic(vec3(0.5f * (1 + Random::get(0.f, 1.f)),
                                                                                 0.5f * (1 + Random::get(0.f, 1.f)),
                                                                                 0.5f * (1 + Random::get(0.f, 1.f))),
                                                                            0.5f * Random::get(0.f, 1.f)),
                                                         is_procedural));
                } else  // Glass
                {
                    models.push_back(
                        Model::createSphere(center, 0.2f, Material::dielectric(1.5f), is_procedural));
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

    return SceneAssets{ std::move(models), std::vector<Texture>(), camera, light};
}

SceneAssets SceneList::teapot() {
    CameraState camera;
    camera.eye = vec3(278, 278, 800);
    camera.look_at = vec3(278, 278, 0);
    camera.up = vec3(0, 1, 0);
    camera.field_of_view = 40;
    camera.aperture = 0.0f;
    camera.focus_distance = 10.0f;
    camera.gamma_correction = true;
    camera.has_sky = false;

    Light light = {
        {213.0f, 553.0f, -328.0f, 0},
        {130.0f, 0.0f, 0.0f, 0},
        {0.0f, 0.0f, 130.0f, 0},
        {0.0f, -1.0f, 0.0f, 0},
        {10.0f, 10.0f, 10.0f, 0},
    };

    const auto i = mat4(1);
    const auto lambertian = Material::lambertian(vec3(0.73f, 0.73f, 0.73f));
    const auto metal = Material::metallic(vec3(0.7f, 0.6f, 0.5f), 0.05f);
    const auto glass = Material::dielectric(1.5f, 0);

    auto box_back = Model::createBox(vec3(0, 1, -165), vec3(165, 330, 0), metal);
    auto teapot = Model::loadModel("../resources/models/teapot.obj");

    teapot.transform(rotate(scale(translate(i, vec3(390, 60, -65)), vec3(5)), radians(-18.0f), vec3(0, 1, 0)));
    box_back.transform(rotate(translate(i, vec3(125, 0, -295)), radians(15.0f), vec3(0, 1, 0)));

    teapot.setMaterial(glass);

    std::vector<Model> models;
    models.push_back(Model::createCornellBox(555));
    models.push_back(teapot);
    models.push_back(box_back);

    return SceneAssets{ std::move(models), std::vector<Texture>(), camera, light};
}

SceneAssets SceneList::cornellBox() {
    CameraState camera;
    camera.eye = vec3(278, 278, 800);
    camera.look_at = vec3(278, 278, 0);
    camera.up = vec3(0, 1, 0);
    camera.field_of_view = 40;
    camera.aperture = 0.0f;
    camera.focus_distance = 10.0f;
    camera.gamma_correction = true;
    camera.has_sky = false;

    Light light = {
        {213.0f, 553.0f, -328.0f, 0},
        {130.0f, 0.0f, 0.0f, 0},
        {0.0f, 0.0f, 130.0f, 0},
        {0.0f, -1.0f, 0.0f, 0},
        {10.0f, 10.0f, 10.0f, 0},
    };
    
    const auto lambertian = Material::lambertian(vec3(0.73f, 0.73f, 0.73f));

    auto box_front = Model::createBox(vec3(0, 1, -165), vec3(165, 165, 0), lambertian);
    auto box_back = Model::createBox(vec3(0, 1, -165), vec3(165, 330, 0), lambertian);

    box_front.transform(rotate(translate(mat4(1), vec3(260, 0, -65)), radians(-18.0f), vec3(0, 1, 0)));
    box_back.transform(rotate(translate(mat4(1), vec3(125, 0, -295)), radians(15.0f), vec3(0, 1, 0)));

    std::vector<Model> models;
    models.push_back(Model::createCornellBox(555));
    models.push_back(box_front);
    models.push_back(box_back);

    return SceneAssets{ std::move(models), std::vector<Texture>(), camera, light};
}

SceneAssets SceneList::cornellBoxDragon() {
    CameraState camera;
    camera.eye = vec3(278, 278, 800);
    camera.look_at = vec3(278, 278, 0);
    camera.up = vec3(0, 1, 0);
    camera.field_of_view = 40;
    camera.aperture = 0.0f;
    camera.focus_distance = 10.0f;
    camera.gamma_correction = true;
    camera.has_sky = false;
    
    Light light = {
        {213.0f, 553.0f, -328.0f, 0},
        {130.0f, 0.0f, 0.0f, 0},
        {0.0f, 0.0f, 130.0f, 0},
        {0.0f, -1.0f, 0.0f, 0},
        {10.0f, 10.0f, 10.0f, 0},
    };

    const auto metal = Material::metallic(vec3(0.9f, 0.4f, 0.0f), 0.85f);
    const auto metal_2 = Material::metallic(vec3(0.0f, 0.67f, 0.8f), 0.55f);

    auto teapot = Model::loadModel("../resources/models/teapot.obj");

    teapot.transform(rotate(scale(translate(mat4(1), vec3(390, 60, -65)), vec3(5)), radians(-18.0f), vec3(0, 1, 0)));
    teapot.setMaterial(metal_2);

    auto dragon = Model::loadModel("../resources/models/dragon.obj");
    dragon.transform(
        rotate(scale(translate(mat4(1), vec3(250, 160, -350)), vec3(600)), radians(135.0f), vec3(0, 1, 0)));
    dragon.setMaterial(metal);

    std::vector<Model> models;
    models.push_back(Model::createCornellBox(555));
    models.push_back(teapot);
    models.push_back(dragon);

    return SceneAssets{ std::move(models), std::vector<Texture>(), camera, light};
}
