#include "material.h"

namespace assets {

Material Material::lambertian(const glm::vec3& albedo, int32_t texture_id) {
    return {
        glm::vec4(albedo, 1),
        glm::vec3(0.0f),
        0.0f,
        glm::vec3(0.0f),
        0.0f,
        0.0f,
        0.0f,
        ShadingModel::Lambertian,
        texture_id,
    };
}

Material Material::metallic(const glm::vec3& albedo, const float roughness, int32_t texture_id) {
    return {
        glm::vec4(albedo, 1),
        glm::vec3(0.0f),
        0.0f,
        glm::vec3(0.0f),
        0.0f,
        0.0f,
        0.0f,
        ShadingModel::Metallic,
        texture_id,
    };
}

Material Material::dielectric(const glm::vec3& albedo, const float refraction_index, int32_t texture_id) {
    return {
        glm::vec4(albedo, 1),
        glm::vec3(0.0f),
        0.0f,
        glm::vec3(0.0f),
        0.0f,
        refraction_index,
        0.0f,
        ShadingModel::Dielectric,
        texture_id,
    };
}

Material Material::emissive(const glm::vec3& albedo, const glm::vec3& emissive, int32_t texture_id) {
    return {
        glm::vec4(albedo, 1),
        emissive,
        0.0f,
        glm::vec3(0.0f),
        0.0f,
        0.0f,
        0.0f,
        ShadingModel::Emissive,
        texture_id,
    };
}

}  // namespace assets
