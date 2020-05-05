const uint MaterialLambertian = 0;
const uint MaterialMetallic = 1;
const uint MaterialDielectric = 2;
const uint MaterialEmissive = 3;

struct Material {
    vec4 albedo;
    vec3 emissive_factor;
    float metallic_factor;
    vec3 specular_factor;
    float roughness_factor;
    float refraction_index;
    float glossiness_factor;
    uint shading_model;
    int albedo_texture;
};