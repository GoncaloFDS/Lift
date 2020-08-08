#pragma once

#include "core/glm.h"

namespace assets {

struct alignas(16) Material {
    // Note: diffuse chance is 1.0f - (specular_chance+refraction_chance)
    glm::vec3  albedo;              // the color used for diffuse lighting
    float specular_chance;      // percentage chance of doing a specular reflection
    glm::vec3  emissive;            // how much the surface glows
    float specular_roughness;   // how rough the specular reflections are
    glm::vec3  specular_color;       // the color tint of specular reflections
    float IOR;                 // index of refraction. used by fresnel and refraction.
    glm::vec3  refraction_color;     // absorption for beer's law
    float refraction_chance;    // percent chance of doing a refractive transmission
    float refraction_roughness; // how rough the refractive transmissions are

    static Material createLambertian(const glm::vec3& color);
    static Material createEmissive(const glm::vec3& color);
};

}  // namespace assets