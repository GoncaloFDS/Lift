#include "random.glsl"
#include "ray_payload.glsl"

float schlick(const float cosine, const float refraction_index) {
    float r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 *= r0;
    return r0 + (1 - r0) * pow(1 - cosine, 5);
}

HitSample lambertian(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, inout uint seed) {
    const bool is_scattered = dot(direction, normal) < 0;
    const vec4 color = mat.albedo_texture >= 0 ? texture(TextureSamplers[mat.albedo_texture], tex_coords) : mat.albedo;
    const vec4 scattered_dir = vec4(normal + RandomInUnitSphere(seed), is_scattered ? 1 : 0);

    return HitSample(color, scattered_dir);
}

HitSample metallic(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, inout uint seed) {
    const vec3 reflected = reflect(direction, normal);
    const bool is_scattered = dot(direction, normal) < 0;

    const vec4 color = mat.albedo_texture >= 0 ? texture(TextureSamplers[mat.albedo_texture], tex_coords) : mat.albedo;
    const vec4 scattered_dir = vec4(reflected + mat.metallic_factor * RandomInUnitSphere(seed), is_scattered ? 1 : 0);

    return HitSample(color, scattered_dir);
}

HitSample dieletric(const Material mat, const vec3 direction, const vec3 normal, const vec2 tex_coords, inout uint seed) {
    const float dot = dot(direction, normal);
    const vec3 out_normal = dot > 0 ? -normal : normal;
    const float ni_over_nt = dot > 0 ? mat.refraction_index : 1 / mat.refraction_index;
    const float cosine = dot > 0 ? mat.refraction_index * dot : -dot;

    const vec3 refracted = refract(direction, out_normal, ni_over_nt);
    const float reflect_prob = refracted != vec3(0) ? schlick(cosine, mat.refraction_index) : 1;

    const vec4 color = mat.albedo_texture >= 0 ? texture(TextureSamplers[mat.albedo_texture], tex_coords) : vec4(1);

    return RandomFloat(seed) < reflect_prob ?
    HitSample(color, vec4(reflect(direction, normal), 1)) :
    HitSample(color, vec4(refracted, 1));
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
        return HitSample(vec4(0.90, 0.00, 0.90, 1), vec4(normalize_dir, 1));
    }
}