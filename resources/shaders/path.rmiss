#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_NV_ray_tracing : require
#include "ray_payload.glsl"
#include "uniform_buffer_object.glsl"

layout(binding = 2) readonly uniform UniformBufferObjectStruct { UniformBufferObject Camera; };

layout(location = 0) rayPayloadInNV PerRayData prd_;

void main() {
//    if (Camera.HasSky) {
//        // Sky color
//        const float t = 0.5*(normalize(gl_WorldRayDirectionNV).y + 1);
//        const vec3 skyColor = mix(vec3(1.0), vec3(0.5, 0.7, 1.0), t);
//
//        Ray.ColorAndDistance = vec4(skyColor, -1);
//    }
//    else {
//        Ray.ColorAndDistance = vec4(0, 0, 0, -1);
//    }
    prd_.radiance = vec3(0.0f);
    prd_.done = 1;
}
