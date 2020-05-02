#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_NV_ray_tracing : require

#include "material.glsl"
#include "uniform_buffer_object.glsl"

layout(binding = 0, set = 0) uniform accelerationStructureNV scene_;
layout(binding = 2) readonly uniform UniformBufferObjectStruct { UniformBufferObject ubo_; };
layout(binding = 3) readonly buffer VertexArray { float Vertices[]; };
layout(binding = 4) readonly buffer IndexArray { uint Indices[]; };
layout(binding = 5) readonly buffer MaterialArray { Material[] Materials; };
layout(binding = 6) readonly buffer OffsetArray { uvec2[] Offsets; };
layout(binding = 7) uniform sampler2D[] TextureSamplers;

#include "scatter.glsl"
#include "vertex.glsl"
#include "sampling.h"

hitAttributeNV vec2 HitAttributes;
layout(location = 0) rayPayloadInNV PerRayData prd_;
layout(location = 2) rayPayloadNV bool shadow_prd_;

const float pi = 3.1415926535897932384626433832795;

struct ParallelogramLight {
    vec3 corner;
    vec3 v1, v2;
    vec3 normal;
    vec3 emission;
};

vec2 Mix(vec2 a, vec2 b, vec2 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

vec3 Mix(vec3 a, vec3 b, vec3 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

void main() {
    // Get the material.
    const uvec2 offsets = Offsets[gl_InstanceCustomIndexNV];
    const uint indexOffset = offsets.x;
    const uint vertexOffset = offsets.y;
    const Vertex v0 = UnpackVertex(vertexOffset + Indices[indexOffset + gl_PrimitiveID * 3 + 0]);
    const Vertex v1 = UnpackVertex(vertexOffset + Indices[indexOffset + gl_PrimitiveID * 3 + 1]);
    const Vertex v2 = UnpackVertex(vertexOffset + Indices[indexOffset + gl_PrimitiveID * 3 + 2]);
    const Material material = Materials[v0.MaterialIndex];

    // Compute the ray hit point properties.
    const vec3 barycentrics = vec3(1.0 - HitAttributes.x - HitAttributes.y, HitAttributes.x, HitAttributes.y);
    const vec3 normal = normalize(Mix(v0.Normal, v1.Normal, v2.Normal, barycentrics));
    const vec2 texCoord = Mix(v0.TexCoord, v1.TexCoord, v2.TexCoord, barycentrics);
    ///////////////////////////////

    // Diffuse hemisphere sampling
    uint seed = prd_.seed;
    const float z1 = radinv_fl(seed, 5 + 3 * prd_.depth);
    const float z2 = radinv_fl(seed, 6 + 3 * prd_.depth);
    vec3 tangent, binormal;
    computeOrthonormalBasis(normal, tangent, binormal);
    vec3 next_sample_dir;
    cosine_sample_hemisphere(z1, z2, next_sample_dir);
    inverse_transform(next_sample_dir, normal, tangent, binormal);
    prd_.direction = next_sample_dir;
    prd_.origin = gl_WorldRayOriginNV + gl_WorldRayDirectionNV * gl_HitTNV;// FIXME

    prd_.attenuation *= material.diffuse.rgb;

    const float lz1 = rnd(seed);
    const float lz2 = rnd(seed);
    prd_.seed = seed;

    ParallelogramLight light;
    light.corner = vec3(213, 553, -328);
    light.v1 = vec3(130, 0, 0);
    light.v2 = vec3(0, 0, 130);
    light.normal = vec3(0, -1, 0);
    light.emission = vec3(15);

    const vec3 light_pos = light.corner + light.v1 * lz1 + light.v2 * lz2;
    vec3 light_dir  = light_pos - prd_.origin;
    const float light_dist = length(light_dir);
    light_dir = normalize(light_dir);
    const float n_dot_l = dot(normal, light_dir);
    const float ln_dot_l = -dot(light.normal, light_dir);

    float weight = 0.0f;

    shadow_prd_ = true;

    if (n_dot_l > 0.0f && ln_dot_l > 0.0f) {
        float tmin = 0.005;
        float tmax = light_dist;

        traceNV(scene_,
        gl_RayFlagsTerminateOnFirstHitNV | gl_RayFlagsOpaqueNV | gl_RayFlagsSkipClosestHitShaderNV,
        0xFF,
        1 /* sbtRecordOffset */,
        0 /* sbtRecordStride */,
        1 /* missIndex */,
        prd_.origin,
        tmin,
        light_dir,
        tmax,
        2 /*payload location*/);

        if (!shadow_prd_) {
            const float A = length(cross(light.v1, light.v2));
            weight = n_dot_l * ln_dot_l * A / (pi * light_dist * light_dist);

        }
    }

    prd_.radiance += light.emission * weight;
}
