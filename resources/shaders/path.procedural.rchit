#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "utils/material.glsl"
#include "utils/uniform_buffer_object.glsl"

layout(binding = 0, set = 0) uniform accelerationStructureEXT scene_;
layout(binding = 2) readonly uniform UniformBufferObjectStruct { UniformBufferObject ubo_; };
layout(binding = 3) readonly buffer VertexArray { float Vertices[]; };
layout(binding = 4) readonly buffer IndexArray { uint Indices[]; };
layout(binding = 5) readonly buffer MaterialArray { Material[] Materials; };
layout(binding = 6) readonly buffer OffsetArray { uvec2[] Offsets; };
layout(binding = 7) uniform sampler2D[] TextureSamplers;
layout(binding = 8) readonly buffer SphereArray { vec4[] Spheres; };

#include "utils/vertex.glsl"
#include "utils/sampling.glsl"
#include "utils/ray_payload.glsl"


hitAttributeEXT vec4 sphere_;
layout(location = 0) rayPayloadInEXT RayPayload ray_;
layout(location = 2) rayPayloadEXT bool shadow_ray_;

const float pi = 3.1415926535897932384626433832795;

vec2 GetSphereTexCoord(const vec3 point) {
    const float phi = atan(point.x, point.z);
    const float theta = asin(point.y);

    return vec2 ((phi + pi) / (2* pi),
    1 - (theta + pi /2) / pi);
}

void main() {
    // Get the material.
    const uvec2 offsets = Offsets[gl_InstanceCustomIndexEXT];
    const uint indexOffset = offsets.x;
    const uint vertexOffset = offsets.y;
    const Vertex v0 = unpackVertex(vertexOffset + Indices[indexOffset]);
    const Material material = Materials[v0.material_index];

    // Compute the ray hit point properties.
    const vec4 sphere = Spheres[gl_InstanceCustomIndexEXT];
    const vec3 center = sphere.xyz;
    const float radius = sphere.w;
    const vec3 point = gl_WorldRayOriginEXT  + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 normal = (point - center) / radius;
    //    const vec3 normal = faceforward(n0, gl_WorldRayDirectionEXT, n0);
    //    if (material.ior <= 0.0f) {
    //        normal = faceforward(normal, gl_WorldRayDirectionEXT, normal);
    //    }
    const vec2 tex_coords = GetSphereTexCoord(normal);
    ///////////////////////////////

    ray_.t = gl_HitTEXT;
    ray_.mat = material;
    ray_.from_inside = dot(normal, gl_WorldRayDirectionEXT) > 0;
    ray_.normal = normal * (ray_.from_inside ? -1.0f : 1.0f);

}
