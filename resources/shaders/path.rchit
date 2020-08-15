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

#include "utils/vertex.glsl"
#include "utils/sampling.glsl"
#include "utils/ray_payload.glsl"

hitAttributeEXT vec2 hit_attributes;
layout(location = 0) rayPayloadInEXT RayPayload ray_;
layout(location = 2) rayPayloadEXT bool shadow_ray_;

const float pi = 3.1415926535897932384626433832795;

vec2 Mix(vec2 a, vec2 b, vec2 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

vec3 Mix(vec3 a, vec3 b, vec3 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

void main() {
    // Compute the ray hit point properties.
    const uvec2 offsets = Offsets[gl_InstanceCustomIndexEXT];
    const uint indexOffset = offsets.x;
    const uint vertexOffset = offsets.y;
    const Vertex v0 = unpackVertex(vertexOffset + Indices[indexOffset + gl_PrimitiveID * 3 + 0]);
    const Vertex v1 = unpackVertex(vertexOffset + Indices[indexOffset + gl_PrimitiveID * 3 + 1]);
    const Vertex v2 = unpackVertex(vertexOffset + Indices[indexOffset + gl_PrimitiveID * 3 + 2]);
    const Material material = Materials[v0.material_index];

    const vec3 barycentrics = vec3(1.0 - hit_attributes.x - hit_attributes.y, hit_attributes.x, hit_attributes.y);
    vec3 normal = normalize(Mix(v0.normal, v1.normal, v2.normal, barycentrics));
//    if (material.refraction_index <= 0.0f) {
//        normal = faceforward(normal, gl_WorldRayDirectionEXT, normal);
//    }
    const vec2 tex_coords = Mix(v0.tex_coords, v1.tex_coords, v2.tex_coords, barycentrics);
    ///////////////////////////////

//    ray_ = scatter(material, gl_WorldRayDirectionEXT, normal, tex_coords, gl_HitTEXT, ray_.seed);
    ray_.t = gl_HitTEXT;
    ray_.mat = material;
//    ray_.mat.albedo = vec4(0.8, 0.8, 0.8, 1);
    ray_.from_inside = dot(normal, gl_WorldRayDirectionEXT) > 0;
    ray_.normal = normal * (ray_.from_inside ? -1.0f : 1.0f);


}
