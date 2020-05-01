#include "Random.glsl"
#include "ray_payload.glsl"

// Polynomial approximation by Christophe Schlick
float Schlick(const float cosine, const float refraction_index) {
    float r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 *= r0;
    return r0 + (1 - r0) * pow(1 - cosine, 5);
}

// lambertian
RayPayload ScatterLambertian(const Material m, const vec3 direction, const vec3 normal, const vec2 texCoord, const float t, inout uint seed) {
    const bool isScattered = dot(direction, normal) < 0;
    const vec4 texColor = m.diffuse_texture >= 0 ? texture(TextureSamplers[m.diffuse_texture], texCoord) : vec4(1);
    const vec4 colorAndDistance = vec4(m.diffuse.rgb * texColor.rgb, t);
    const vec4 scatter = vec4(normal + RandomInUnitSphere(seed), isScattered ? 1 : 0);

    return RayPayload(colorAndDistance, scatter, seed);
}

// metallic
RayPayload ScatterMetallic(const Material m, const vec3 direction, const vec3 normal, const vec2 texCoord, const float t, inout uint seed) {
    const vec3 reflected = reflect(direction, normal);
    const bool isScattered = dot(reflected, normal) > 0;

    const vec4 texColor = m.diffuse_texture >= 0 ? texture(TextureSamplers[m.diffuse_texture], texCoord) : vec4(1);
    const vec4 colorAndDistance = isScattered ? vec4(m.diffuse.rgb * texColor.rgb, t) : vec4(1, 1, 1, -1);
    const vec4 scatter = vec4(reflected + m.metallic_factor*RandomInUnitSphere(seed), isScattered ? 1 : 0);

    return RayPayload(colorAndDistance, scatter, seed);
}

// dielectric
RayPayload ScatterDieletric(const Material m, const vec3 direction, const vec3 normal, const vec2 texCoord, const float t, inout uint seed) {
    const float dot = dot(direction, normal);
    const vec3 outwardNormal = dot > 0 ? -normal : normal;
    const float niOverNt = dot > 0 ? m.refraction_index : 1 / m.refraction_index;
    const float cosine = dot > 0 ? m.refraction_index * dot : -dot;

    const vec3 refracted = refract(direction, outwardNormal, niOverNt);
    const float reflectProb = refracted != vec3(0) ? Schlick(cosine, m.refraction_index) : 1;

    const vec4 texColor = m.diffuse_texture >= 0 ? texture(TextureSamplers[m.diffuse_texture], texCoord) : vec4(1);

    return RandomFloat(seed) < reflectProb
    ? RayPayload(vec4(texColor.rgb, t), vec4(reflect(direction, normal), 1), seed)
    : RayPayload(vec4(texColor.rgb, t), vec4(refracted, 1), seed);
}

// Diffuse Light
RayPayload ScatterDiffuseLight(const Material m, const float t, inout uint seed) {
    const vec4 colorAndDistance = vec4(m.diffuse.rgb, t);
    const vec4 scatter = vec4(1, 0, 0, 0);

    return RayPayload(colorAndDistance, scatter, seed);
}

RayPayload Scatter(const Material m, const vec3 direction, const vec3 normal, const vec2 texCoord, const float t, inout uint seed) {
    const vec3 normDirection = normalize(direction);

    switch (m.shading_model) {
        case MaterialLambertian:
			return ScatterLambertian(m, normDirection, normal, texCoord, t, seed);
        case MaterialMetallic:
			return ScatterMetallic(m, normDirection, normal, texCoord, t, seed);
        case MaterialDielectric:
			return ScatterDieletric(m, normDirection, normal, texCoord, t, seed);
        case MaterialDiffuseLight:
			return ScatterDiffuseLight(m, t, seed);
    }
}

