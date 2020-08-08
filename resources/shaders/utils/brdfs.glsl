#include "random.glsl"
#include "utils/ray_payload.glsl"

#define M_PI 3.1415926535897932384626433832795

float schlick(const float cosine, const float refraction_index) {
    float r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 *= r0;
    return r0 + (1 - r0) * pow(1 - cosine, 5);
}

RayPayload lambertian(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, const float t, inout uint seed) {
    const bool is_scattered = dot(direction, normal) < 0;
    const vec4 color = mat.albedo_texture >= 0 ? texture(TextureSamplers[mat.albedo_texture], tex_coords) : mat.albedo;
    const vec3 scattered_dir = normal + randomInUnitSphere(seed);
    const float pdf = dot(normal, scattered_dir.xyz) / M_PI;

    RayPayload ray;
    ray.color = color.rgb;
    ray.t = t;
    ray.scattered_dir = scattered_dir;
    ray.is_scattered = is_scattered;
    ray.seed = seed;
    ray.pdf = pdf;
    return ray;
}

RayPayload metallic(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, const float t, inout uint seed) {
    const vec3 reflected = reflect(direction, normal);
    const bool is_scattered = dot(reflected, normal) > 0;

    const vec4 color = mat.albedo_texture >= 0 ? texture(TextureSamplers[mat.albedo_texture], tex_coords) : mat.albedo;
    const vec3 scattered_dir = reflected + mat.metallic_factor * randomInUnitSphere(seed);

    const float pdf = 1;

    RayPayload ray;
    ray.color = is_scattered ? color.rgb : vec3(1);
    ray.t = is_scattered ? t : -1;
    ray.scattered_dir = scattered_dir;
    ray.is_scattered = is_scattered;
    ray.seed = seed;
    ray.pdf = pdf;
    return ray;
}

RayPayload dieletric(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, const float t, inout uint seed) {
    const float dot = dot(direction, normal);
    const vec3 out_normal = dot > 0 ? -normal : normal;
    const float ni_over_nt = dot > 0 ? mat.refraction_index : 1 / mat.refraction_index;
    const float cosine = dot > 0 ? mat.refraction_index * dot : -dot;

    const vec3 refracted = refract(direction, out_normal, ni_over_nt);
    const float reflect_prob = refracted != vec3(0) ? schlick(cosine, mat.refraction_index) : 1;

    const vec4 color = mat.albedo_texture >= 0 ? texture(TextureSamplers[mat.albedo_texture], tex_coords) : vec4(1);

    const float pdf = 1;

    const vec3 scattered_dir = randomFloat(seed) < reflect_prob ? reflect(direction, normal) : refracted;

    RayPayload ray;
    ray.color = color.rgb;
    ray.t = t;
    ray.scattered_dir = scattered_dir;
    ray.is_scattered = true;
    ray.seed = seed;
    ray.pdf = pdf;
    return ray;
}

RayPayload emissive(const Material mat, const float t, inout uint seed) {
    const float pdf = 1;

    RayPayload ray;
    ray.color = mat.albedo.rgb;
    ray.t = t;
    ray.scattered_dir = vec3(1, 0, 0);
    ray.is_scattered = false;
    ray.seed = seed;
    ray.pdf = pdf;
    return ray;
}

RayPayload scatter(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, const float t, inout uint seed) {
    const vec3 normalize_dir = normalize(direction);

    switch (mat.shading_model) {
        case MaterialLambertian:
            return lambertian(mat, normalize_dir, normal, tex_coords, t, seed);
        case MaterialMetallic:
            return metallic(mat, normalize_dir, normal, tex_coords, t, seed);
        case MaterialDielectric:
            return dieletric(mat, normalize_dir, normal, tex_coords, t, seed);
        case MaterialEmissive:
            return emissive(mat, t, seed);
    }
}