#pragma once

#include "core/glm.h"

namespace assets {

struct alignas(16) Material final {
    static Material lambertian(const glm::vec3 &diffuse, const int32_t texture_id = -1) {
        return Material {glm::vec4(diffuse, 1), texture_id, 0.0f, 0.0f, Enum::Lambertian};
    }

    static Material metallic(const glm::vec3 &diffuse, const float fuzziness, const int32_t texture_id = -1) {
        return Material {glm::vec4(diffuse, 1), texture_id, fuzziness, 0.0f, Enum::Metallic};
    }

    static Material dielectric(const float refraction_index, const int32_t texture_id = -1) {
        return Material {glm::vec4(0.7f, 0.7f, 1.0f, 1), texture_id, 0.0f, refraction_index, Enum::Dielectric};
    }

    static Material isotropic(const glm::vec3 &diffuse, const int32_t texture_id = -1) {
        return Material {glm::vec4(diffuse, 1), texture_id, 0.0f, 0.0f, Enum::Isotropic};
    }

    static Material diffuseLight(const glm::vec3 &diffuse, const int32_t texture_id = -1) {
        return Material {glm::vec4(diffuse, 1), texture_id, 0.0f, 0.0f, Enum::DiffuseLight};
    }

    enum class Enum : uint32_t { Lambertian = 0, Metallic = 1, Dielectric = 2, Isotropic = 3, DiffuseLight = 4 };

    // Note: vec3 and vec4 gets aligned on 16 bytes in vulkan shaders.

    // Base material
    glm::vec4 diffuse;
    int32_t diffuseTextureId;

    // Metal fuzziness
    float fuzziness;

    // dielectric refraction index
    float refractionIndex;

    // Which material are we dealing with
    Enum materialModel;
};

}  // namespace assets