#include "scene_list.h"
#include "assets/material.h"
#include "assets/model.h"
#include "assets/texture.h"
#include <functional>
#include <random>

using namespace glm;
using assets::Material;
using assets::Model;
using assets::Texture;

const std::vector<std::pair<std::string, std::function<SceneAssets(SceneList::CameraInitialSate&)>>>
    SceneList::allScenes = {
    {"Cube And Spheres", cubeAndSpheres},
    {"Ray Tracing In One Weekend", rayTracingInOneWeekend},
    {"Planets In One Weekend", planetsInOneWeekend},
    {"Lucy In One Weekend", lucyInOneWeekend},
    {"Cornell Box", cornellBox},
    {"Cornell Box & Lucy", cornellBoxLucy},
};

SceneAssets SceneList::cubeAndSpheres(CameraInitialSate& camera) {
    // Basic test scene.

    camera.modelView = translate(mat4(1), vec3(0, 0, -2));
    camera.fieldOfView = 90;
    camera.aperture = 0.05f;
    camera.focusDistance = 2.0f;
    camera.gammaCorrection = false;
    camera.hasSky = true;

    std::vector<Model> models;
    std::vector<Texture> textures;

    models.push_back(Model::LoadModel("../resources/models/cube_multi.obj"));
    models.push_back(Model::CreateSphere(vec3(1, 0, 0), 0.5, Material::metallic(vec3(0.7f, 0.5f, 0.8f), 0.2f), true));
    models.push_back(Model::CreateSphere(vec3(-1, 0, 0), 0.5, Material::dielectric(1.5f), true));
    models.push_back(Model::CreateSphere(vec3(0, 1, 0), 0.5, Material::lambertian(vec3(1.0f), 0), true));

    textures.push_back(Texture::LoadTexture("../resources/textures/land_ocean_ice_cloud_2048.png",
                                            vulkan::SamplerConfig()));

    return std::forward_as_tuple(std::move(models), std::move(textures));
}

SceneAssets SceneList::rayTracingInOneWeekend(CameraInitialSate& camera) {
    // Final scene from Ray Tracing In One Weekend book.

    camera.modelView = lookAt(vec3(13, 2, 3), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.fieldOfView = 20;
    camera.aperture = 0.1f;
    camera.focusDistance = 10.0f;
    camera.gammaCorrection = true;
    camera.hasSky = true;

    const bool isProc = true;

    std::mt19937 engine(42);
    auto random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<Model> models;

    models.push_back(Model::CreateSphere(vec3(0, -1000, 0),
                                         1000,
                                         Material::lambertian(vec3(0.5f, 0.5f, 0.5f)),
                                         isProc));

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            const float choose_mat = random();
            const vec3 center(a + 0.9f * random(), 0.2f, b + 0.9f * random());

            if (length(center - vec3(4, 0.2f, 0)) > 0.9) {
                if (choose_mat < 0.8f) // Diffuse
                {
                    models.push_back(Model::CreateSphere(center, 0.2f, Material::lambertian(vec3(
                        random() * random(),
                        random() * random(),
                        random() * random())),
                                                         isProc));
                } else if (choose_mat < 0.95f) // Metal
                {
                    models.push_back(Model::CreateSphere(center, 0.2f, Material::metallic(
                        vec3(0.5f * (1 + random()), 0.5f * (1 + random()), 0.5f * (1 + random())),
                        0.5f * random()),
                                                         isProc));
                } else // Glass
                {
                    models.push_back(Model::CreateSphere(center, 0.2f, Material::dielectric(1.5f), isProc));
                }
            }
        }
    }

    models.push_back(Model::CreateSphere(vec3(0, 1, 0), 1.0f, Material::dielectric(1.5f), isProc));
    models.push_back(Model::CreateSphere(vec3(-4, 1, 0), 1.0f, Material::lambertian(vec3(0.4f, 0.2f, 0.1f)), isProc));
    models.push_back(Model::CreateSphere(vec3(4, 1, 0),
                                         1.0f,
                                         Material::metallic(vec3(0.7f, 0.6f, 0.5f), 0.0f),
                                         isProc));

    return std::forward_as_tuple(std::move(models), std::vector<Texture>());
}

SceneAssets SceneList::planetsInOneWeekend(CameraInitialSate& camera) {
    // Same as RayTracingInOneWeekend but using textures.

    camera.modelView = lookAt(vec3(13, 2, 3), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.fieldOfView = 20;
    camera.aperture = 0.1f;
    camera.focusDistance = 10.0f;
    camera.gammaCorrection = true;
    camera.hasSky = true;

    const bool is_proc = true;

    std::mt19937 engine(42);
    auto random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<Model> models;
    std::vector<Texture> textures;

    models.push_back(Model::CreateSphere(vec3(0, -1000, 0),
                                         1000,
                                         Material::lambertian(vec3(0.5f, 0.5f, 0.5f)),
                                         is_proc));

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            const float choose_mat = random();
            const vec3 center(a + 0.9f * random(), 0.2f, b + 0.9f * random());

            if (length(center - vec3(4, 0.2f, 0)) > 0.9) {
                if (choose_mat < 0.8f) // Diffuse
                {
                    models.push_back(Model::CreateSphere(center, 0.2f, Material::lambertian(vec3(
                        random() * random(),
                        random() * random(),
                        random() * random())), is_proc));
                } else if (choose_mat < 0.95f) // Metal
                {
                    models.push_back(Model::CreateSphere(center, 0.2f, Material::metallic(
                        vec3(0.5f * (1 + random()), 0.5f * (1 + random()), 0.5f * (1 + random())),
                        0.5f * random()), is_proc));
                } else // Glass
                {
                    models.push_back(Model::CreateSphere(center, 0.2f, Material::dielectric(1.5f), is_proc));
                }
            }
        }
    }

    models.push_back(Model::CreateSphere(vec3(0, 1, 0), 1.0f, Material::metallic(vec3(1.0f), 0.1f, 2), is_proc));
    models.push_back(Model::CreateSphere(vec3(-4, 1, 0), 1.0f, Material::lambertian(vec3(1.0f), 0), is_proc));
    models.push_back(Model::CreateSphere(vec3(4, 1, 0), 1.0f, Material::metallic(vec3(1.0f), 0.0f, 1), is_proc));

    textures.push_back(Texture::LoadTexture("../resources/textures/2k_mars.jpg", vulkan::SamplerConfig()));
    textures.push_back(Texture::LoadTexture("../resources/textures/2k_moon.jpg", vulkan::SamplerConfig()));
    textures.push_back(Texture::LoadTexture("../resources/textures/land_ocean_ice_cloud_2048.png",
                                            vulkan::SamplerConfig()));

    return std::forward_as_tuple(std::move(models), std::move(textures));
}

SceneAssets SceneList::lucyInOneWeekend(CameraInitialSate& camera) {
    // Same as RayTracingInOneWeekend but using the Lucy 3D model.

    camera.modelView = lookAt(vec3(13, 2, 3), vec3(0, 1.0, 0), vec3(0, 1, 0));
    camera.fieldOfView = 20;
    camera.aperture = 0.05f;
    camera.focusDistance = 10.0f;
    camera.gammaCorrection = true;
    camera.hasSky = true;

    const bool is_proc = true;

    std::mt19937 engine(42);
    auto random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<Model> models;

    models.push_back(Model::CreateSphere(vec3(0, -1000, 0),
                                         1000,
                                         Material::lambertian(vec3(0.5f, 0.5f, 0.5f)),
                                         is_proc));

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            const float choose_mat = random();
            const vec3 center(a + 0.9f * random(), 0.2f, b + 0.9f * random());

            if (length(center - vec3(4, 0.2f, 0)) > 0.9) {
                if (choose_mat < 0.8f) // Diffuse
                {
                    models.push_back(Model::CreateSphere(center, 0.2f, Material::lambertian(vec3(
                        random() * random(),
                        random() * random(),
                        random() * random())), is_proc));
                } else if (choose_mat < 0.95f) // Metal
                {
                    models.push_back(Model::CreateSphere(center, 0.2f, Material::metallic(
                        vec3(0.5f * (1 + random()), 0.5f * (1 + random()), 0.5f * (1 + random())),
                        0.5f * random()), is_proc));
                } else // Glass
                {
                    models.push_back(Model::CreateSphere(center, 0.2f, Material::dielectric(1.5f), is_proc));
                }
            }
        }
    }

    auto lucy_0 = Model::LoadModel("../resources/models/lucy.obj");
    auto lucy_1 = lucy_0;
    auto lucy_2 = lucy_0;

    const auto i = mat4(1);
    const float scale_factor = 0.0035f;

    lucy_0.Transform(rotate(scale(translate(i, vec3(0, -0.08f, 0)), vec3(scale_factor)),
                            radians(90.0f), vec3(0, 1, 0)));

    lucy_1.Transform(rotate(scale(translate(i, vec3(-4, -0.08f, 0)), vec3(scale_factor)),
                            radians(90.0f), vec3(0, 1, 0)));

    lucy_2.Transform(rotate(scale(translate(i, vec3(4, -0.08f, 0)), vec3(scale_factor)),
                            radians(90.0f), vec3(0, 1, 0)));

    lucy_0.SetMaterial(Material::dielectric(1.5f));
    lucy_1.SetMaterial(Material::lambertian(vec3(0.4f, 0.2f, 0.1f)));
    lucy_2.SetMaterial(Material::metallic(vec3(0.7f, 0.6f, 0.5f), 0.05f));

    models.push_back(std::move(lucy_0));
    models.push_back(std::move(lucy_1));
    models.push_back(std::move(lucy_2));

    return std::forward_as_tuple(std::move(models), std::vector<Texture>());
}

SceneAssets SceneList::cornellBox(CameraInitialSate& camera) {
    camera.modelView = lookAt(vec3(278, 278, 800), vec3(278, 278, 0), vec3(0, 1, 0));
    camera.fieldOfView = 40;
    camera.aperture = 0.0f;
    camera.focusDistance = 10.0f;
    camera.gammaCorrection = true;
    camera.hasSky = false;

    const auto i = mat4(1);
    const auto white = Material::lambertian(vec3(0.73f, 0.73f, 0.73f));

    auto box_0 = Model::CreateBox(vec3(0, 0, -165), vec3(165, 165, 0), white);
    auto box_1 = Model::CreateBox(vec3(0, 0, -165), vec3(165, 330, 0), white);

    box_0.Transform(rotate(translate(i, vec3(555 - 130 - 165, 0, -65)), radians(-18.0f), vec3(0, 1, 0)));
    box_1.Transform(rotate(translate(i, vec3(555 - 265 - 165, 0, -295)), radians(15.0f), vec3(0, 1, 0)));

    std::vector<Model> models;
    models.push_back(Model::CreateCornellBox(555));
    models.push_back(box_0);
    models.push_back(box_1);

    return std::make_tuple(std::move(models), std::vector<Texture>());
}

SceneAssets SceneList::cornellBoxLucy(CameraInitialSate& camera) {
    camera.modelView = lookAt(vec3(278, 278, 800), vec3(278, 278, 0), vec3(0, 1, 0));
    camera.fieldOfView = 40;
    camera.aperture = 0.0f;
    camera.focusDistance = 10.0f;
    camera.gammaCorrection = true;
    camera.hasSky = false;

    const auto i = mat4(1);
    const auto sphere =
        Model::CreateSphere(vec3(555 - 130, 165.0f, -165.0f / 2 - 65), 80.0f, Material::dielectric(1.5f), true);
    auto lucy_0 = Model::LoadModel("../resources/models/lucy.obj");

    lucy_0.Transform(rotate(scale(translate(i, vec3(555 - 300 - 165 / 2, -9, -295 - 165 / 2)), vec3(0.6f)),
                            radians(75.0f), vec3(0, 1, 0)));

    std::vector<Model> models;
    models.push_back(Model::CreateCornellBox(555));
    models.push_back(sphere);
    models.push_back(lucy_0);

    return std::forward_as_tuple(std::move(models), std::vector<Texture>());
}
