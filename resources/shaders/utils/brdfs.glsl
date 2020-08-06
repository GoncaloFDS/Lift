#include "random.glsl"

#define M_PI 3.1415926535897932384626433832795

struct HitSample {
    vec4 color;// rgb + t
    vec4 scattered_dir;// xyz + w (is scatter needed)
    bool done;
    float pdf;
};

float schlick(const float cosine, const float refraction_index) {
    float r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 *= r0;
    return r0 + (1 - r0) * pow(1 - cosine, 5);
}

HitSample lambertian(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, inout uint seed) {
    const bool is_scattered = dot(direction, normal) < 0;
    const vec4 color = mat.albedo_texture >= 0 ? texture(TextureSamplers[mat.albedo_texture], tex_coords) : mat.albedo;
    const vec4 scattered_dir = vec4(normal + randomInUnitSphere(seed), is_scattered ? 1 : 0);
    const float pdf = dot(normal, scattered_dir.xyz) / M_PI;

    return HitSample(color, scattered_dir, false, pdf);
}

HitSample metallic(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, inout uint seed) {
    const vec3 reflected = reflect(direction, normal);
    const bool is_scattered = dot(direction, normal) < 0;

    const vec4 color = mat.albedo_texture >= 0 ? texture(TextureSamplers[mat.albedo_texture], tex_coords) : mat.albedo;
    const vec4 scattered_dir = vec4(reflected + mat.metallic_factor * randomInUnitSphere(seed), is_scattered ? 1 : 0);

    const float pdf = 1;
    return HitSample(color, scattered_dir, false, pdf);
}

HitSample dieletric(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, inout uint seed) {
    const float dot = dot(direction, normal);
    const vec3 out_normal = dot > 0 ? -normal : normal;
    const float ni_over_nt = dot > 0 ? mat.refraction_index : 1 / mat.refraction_index;
    const float cosine = dot > 0 ? mat.refraction_index * dot : -dot;

    const vec3 refracted = refract(direction, out_normal, ni_over_nt);
    const float reflect_prob = refracted != vec3(0) ? schlick(cosine, mat.refraction_index) : 1;

    const vec4 color = mat.albedo_texture >= 0 ? texture(TextureSamplers[mat.albedo_texture], tex_coords) : vec4(1);

    const float pdf = 1;
    return randomFloat(seed) < reflect_prob ?
    HitSample(color, vec4(reflect(direction, normal), 1), false, pdf) :
    HitSample(color, vec4(refracted, 1), false, pdf);
}

HitSample emissive(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, inout uint seed) {
    const float pdf = 1;
    return HitSample(mat.albedo, vec4(direction, 1), true, pdf);
}

HitSample scatter(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, inout uint seed) {
    const vec3 normalize_dir = normalize(direction);

    switch (mat.shading_model) {
        case MaterialLambertian:
            return lambertian(mat, normalize_dir, normal, tex_coords, seed);
        case MaterialMetallic:
            return metallic(mat, normalize_dir, normal, tex_coords, seed);
        case MaterialDielectric:
            return dieletric(mat, normalize_dir, normal, tex_coords, seed);
        case MaterialEmissive:
            return emissive(mat, normalize_dir, normal, tex_coords, seed);
//        return HitSample(mat.albedo, vec4(normalize_dir, 1), true, pdf);
    }
}