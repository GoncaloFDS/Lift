#pragma once

#include "core/glm.h"

namespace assets {

enum class ShadingModel : uint32_t { Lambertian = 0, Metallic = 1, Dielectric = 2, Isotropic = 3, Emissive = 4 };
enum class AlphaMode : uint32_t { Opaque = 0, Mask = 1, Blend = 2 };

struct Material {
    glm::vec4 albedo;
    glm::vec3 emissive_factor;
    float metallic_factor;
    glm::vec3 specular_factor;
    float roughness_factor;
    float refraction_index;
    float glossiness_factor;
    ShadingModel shading_model;
    int32_t albedo_texture;

    // Default constructors

    static Material lambertian(const glm::vec3& albedo, int32_t texture_id = -1);
    static Material metallic(const glm::vec3& albedo, float roughness, int32_t texture_id = -1);
    static Material dielectric(const glm::vec3& albedo, float refraction_index, int32_t texture_id = -1);
    static Material emissive(const glm::vec3& albedo, int32_t texture_id = -1);
};

}  // namespace assets