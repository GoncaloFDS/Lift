#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "utils/material.glsl"
#include "utils/uniform_buffer_object.glsl"
#include "utils/random.glsl"
#include "utils/ray_payload.glsl"

layout(binding = 0, set = 0) uniform accelerationStructureEXT scene_;
layout(binding = 3) readonly uniform UniformBufferObjectStruct { UniformBufferObject ubo_; };
layout(binding = 4) readonly buffer VertexArray { float[] vertices_; };
layout(binding = 5) readonly buffer IndexArray { uint[] indices_; };
layout(binding = 6) readonly buffer MaterialArray { Material[] materials_; };
layout(binding = 8) readonly buffer OffsetArray { uvec2[] offsets_; };
layout(binding = 9) uniform sampler2D[] textures_;
layout(binding = 10) readonly buffer SphereArray { vec4[] spheres_; };

#include "utils/vertex.glsl"

hitAttributeEXT vec4 sphere_;

layout(location = 0) rayPayloadInEXT RayPayload ray_;
layout(location = 2) rayPayloadEXT bool shadow_ray_;

vec2 GetSphereTexCoord(const vec3 point) {
    const float phi = atan(point.x, point.z);
    const float theta = asin(point.y);

    return vec2 ((phi + c_pi) / (2* c_pi),
    1 - (theta + c_pi /2) / c_pi);
}

void main() {
    // Get the material.
    const uvec2 offsets = offsets_[gl_InstanceCustomIndexEXT];
    const uint indexOffset = offsets.x;
    const uint vertexOffset = offsets.y;
    const Vertex v0 = unpackVertex(vertexOffset + indices_[indexOffset]);
    const Material material = materials_[v0.material_index];

    // Compute the ray hit point properties.
    const vec4 sphere = spheres_[gl_InstanceCustomIndexEXT];
    const vec3 center = sphere.xyz;
    const float radius = sphere.w;
    const vec3 point = gl_WorldRayOriginEXT  + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 normal = (point - center) / radius;
    const vec2 tex_coords = GetSphereTexCoord(normal);
    ///////////////////////////////

    ray_.t = gl_HitTEXT;
    ray_.mat = material;
    ray_.from_inside = dot(normal, gl_WorldRayDirectionEXT) > 0;
    ray_.normal = normal * (ray_.from_inside ? -1.0f : 1.0f);

}
