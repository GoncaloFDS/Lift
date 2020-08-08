#include "material.h"

namespace assets {

using namespace glm;

Material Material::createLambertian(const vec3& color) {
    Material mat = {};
    mat.albedo = color;
    mat.emissive = vec3(0.0f);
    mat.specular_chance = 0.0f;
    mat.specular_roughness = 0.0f;
    mat.specular_color = vec3(0.0f);
    mat.IOR = 1.0f;
    mat.refraction_chance = 0.0f;
    mat.refraction_roughness = 0.0f;
    mat.refraction_color = vec3(0.0f);
    return mat;
}
Material Material::createEmissive(const vec3& color) {
    Material mat = {};
    mat.albedo = vec3(0.0f);
    mat.emissive = color;
    mat.specular_chance = 0.0f;
    mat.specular_roughness = 0.0f;
    mat.specular_color = vec3(0.0f);
    mat.IOR = 1.0f;
    mat.refraction_chance = 0.0f;
    mat.refraction_roughness = 0.0f;
    mat.refraction_color = vec3(0.0f);
    return mat;
}

}  // namespace assets
