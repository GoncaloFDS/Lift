#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_NV_ray_tracing : require

#include "utils/material.glsl"
#include "utils/uniform_buffer_object.glsl"

layout(binding = 0, set = 0) uniform accelerationStructureNV scene_;
layout(binding = 2) readonly uniform UniformBufferObjectStruct { UniformBufferObject ubo_; };
layout(binding = 3) readonly buffer VertexArray { float Vertices[]; };
layout(binding = 4) readonly buffer IndexArray { uint Indices[]; };
layout(binding = 5) readonly buffer MaterialArray { Material[] Materials; };
layout(binding = 6) readonly buffer OffsetArray { uvec2[] Offsets; };
layout(binding = 7) uniform sampler2D[] TextureSamplers;
layout(binding = 8) readonly buffer SphereArray { vec4[] Spheres; };

#include "utils/brdfs.glsl"
#include "utils/vertex.glsl"
#include "utils/sampling.h"

hitAttributeNV vec4 sphere_;
layout(location = 0) rayPayloadInNV PerRayData prd_;
layout(location = 2) rayPayloadNV bool shadow_prd_;

const float pi = 3.1415926535897932384626433832795;

vec2 GetSphereTexCoord(const vec3 point) {
    const float phi = atan(point.x, point.z);
    const float theta = asin(point.y);
    const float pi = 3.1415926535897932384626433832795;

    return vec2 ((phi + pi) / (2* pi),
    1 - (theta + pi /2) / pi);
}

void main() {
    // Get the material.
    const uvec2 offsets = Offsets[gl_InstanceCustomIndexNV];
    const uint indexOffset = offsets.x;
    const uint vertexOffset = offsets.y;
    const Vertex v0 = unpackVertex(vertexOffset + Indices[indexOffset]);
    const Material material = Materials[v0.material_index];

    // Compute the ray hit point properties.
    const vec4 sphere = Spheres[gl_InstanceCustomIndexNV];
    const vec3 center = sphere.xyz;
    const float radius = sphere.w;
    const vec3 point = gl_WorldRayOriginNV + gl_HitTNV * gl_WorldRayDirectionNV;
    vec3 normal = (point - center) / radius;
    //    const vec3 normal = faceforward(n0, gl_WorldRayDirectionNV, n0);
    if (material.refraction_index <= 0.0f) {
        normal = faceforward(normal, gl_WorldRayDirectionNV, normal);
    }
    const vec2 tex_coords = GetSphereTexCoord(normal);

    // Diffuse hemisphere sampling
    uint seed = prd_.seed;

    HitSample hit = scatter(material, gl_WorldRayDirectionNV, normal, tex_coords, prd_.seed);

    prd_.direction = hit.scattered_dir.xyz;
    prd_.origin = gl_WorldRayOriginNV + gl_WorldRayDirectionNV * gl_HitTNV;
    prd_.attenuation *= hit.color.xyz;

    const float lz1 = rnd(seed);
    const float lz2 = rnd(seed);
    prd_.seed = seed;

    // MIS
    ParallelogramLight light = ubo_.light;

    const vec3 light_pos = light.corner.xyz + light.v1.xyz * lz1 + light.v2.xyz * lz2;
    vec3 light_dir  = light_pos - prd_.origin;
    const float light_dist = length(light_dir);
    light_dir = normalize(light_dir);
    const float n_dot_l = dot(normal, light_dir);
    const float ln_dot_l = -dot(light.normal.xyz, light_dir);

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
            const float A = length(cross(light.v1.xyz, light.v2.xyz));
            weight = n_dot_l * ln_dot_l * A / (pi * light_dist * light_dist);

        }
    }

    prd_.radiance += light.emission.xyz * weight;

}
