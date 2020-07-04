#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_NV_ray_tracing : require

#include "utils/material.glsl"
#include "utils/uniform_buffer_object.glsl"
#include "utils/ray_payload.glsl"

layout(binding = 0, set = 0) uniform accelerationStructureNV scene_;
layout(binding = 2) readonly uniform UniformBufferObjectStruct { UniformBufferObject ubo_; };
layout(binding = 3) readonly buffer VertexArray { float Vertices[]; };
layout(binding = 4) readonly buffer IndexArray { uint Indices[]; };
layout(binding = 5) readonly buffer MaterialArray { Material[] Materials; };
layout(binding = 6) readonly buffer OffsetArray { uvec2[] Offsets; };
layout(binding = 7) uniform sampler2D[] TextureSamplers;
layout(binding = 8) buffer LightNodes { PathNode[] light_nodes_; };
layout(binding = 9) buffer CameraNodes { PathNode[] camera_nodes_; };

#include "utils/brdfs.glsl"
#include "utils/vertex.glsl"
#include "utils/sampling.glsl"

hitAttributeNV vec2 hit_attributes;
layout(location = 0) rayPayloadInNV PerRayData prd_;
layout(location = 2) rayPayloadNV bool shadow_prd_;

const float pi = 3.1415926535897932384626433832795;

vec2 Mix(vec2 a, vec2 b, vec2 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

vec3 Mix(vec3 a, vec3 b, vec3 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

void main() {
    // Compute the ray hit point properties.
    const uvec2 offsets = Offsets[gl_InstanceCustomIndexNV];
    const uint indexOffset = offsets.x;
    const uint vertexOffset = offsets.y;
    const Vertex v0 = unpackVertex(vertexOffset + Indices[indexOffset + gl_PrimitiveID * 3 + 0]);
    const Vertex v1 = unpackVertex(vertexOffset + Indices[indexOffset + gl_PrimitiveID * 3 + 1]);
    const Vertex v2 = unpackVertex(vertexOffset + Indices[indexOffset + gl_PrimitiveID * 3 + 2]);
    const Material material = Materials[v0.material_index];

    const vec3 barycentrics = vec3(1.0 - hit_attributes.x - hit_attributes.y, hit_attributes.x, hit_attributes.y);
    vec3 normal = normalize(Mix(v0.normal, v1.normal, v2.normal, barycentrics));
    if (material.refraction_index <= 0.0f) {
        normal = faceforward(normal, gl_WorldRayDirectionNV, normal);
    }
    const vec2 tex_coords = Mix(v0.tex_coords, v1.tex_coords, v2.tex_coords, barycentrics);
    ///////////////////////////////


    // Diffuse hemisphere sampling
    uint seed = prd_.seed;

    HitSample hit = scatter(material, gl_WorldRayDirectionNV, normal, tex_coords, prd_.seed);

    prd_.direction = hit.scattered_dir.xyz;
    prd_.origin = gl_WorldRayOriginNV + gl_WorldRayDirectionNV * gl_HitTNV;
    prd_.attenuation *= hit.color.xyz;

    if (hit.done) {
        prd_.done = hit.done;
        prd_.radiance = hit.color.xyz;
        return;
    }

    const float lz1 = rnd(seed);
    const float lz2 = rnd(seed);
    prd_.seed = seed;

    // MIS

    const vec3 light_pos = light_nodes_[0].position.xyz;
    vec3 light_dir  = light_pos - prd_.origin;
    const float light_dist = length(light_dir);
    light_dir = normalize(light_dir);
    const float n_dot_l = dot(normal, light_dir);
    const float ln_dot_l = -dot(light_nodes_[0].normal.xyz, light_dir);

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
//            const float A = length(cross(light.v1.xyz, light.v2.xyz));
            const float A = 1000;
            weight = n_dot_l * ln_dot_l * A / (pi * light_dist * light_dist);
        }
    }

//    prd_.radiance += light.emission.xyz * weight;
}
