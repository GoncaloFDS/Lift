#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_NV_ray_tracing : require
#include "material.glsl"

layout(binding = 3) readonly buffer VertexArray { float Vertices[]; };
layout(binding = 4) readonly buffer IndexArray { uint Indices[]; };
layout(binding = 5) readonly buffer MaterialArray { Material[] Materials; };
layout(binding = 6) readonly buffer OffsetArray { uvec2[] Offsets; };
layout(binding = 7) uniform sampler2D[] TextureSamplers;

#include "scatter.glsl"
#include "vertex.glsl"

hitAttributeNV vec2 HitAttributes;
rayPayloadInNV PerRayData prd_;

const float pi = 3.1415926535897932384626433832795;

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

	prd_.origin = gl_WorldRayOriginNV + gl_WorldRayDirectionNV * gl_HitTNV; // FIXME

	vec3 diffuse_color = material.diffuse.rgb;
	prd_.attenuation = prd_.attenuation * diffuse_color / pi;

	if (material.shading_model == MaterialDiffuseLight) {
		prd_.radiance = material.diffuse.rgb;
		prd_.done = 1;
		return;
	}

	prd_.direction = normal + RandomInUnitSphere(prd_.seed);
	prd_.radiance = vec3(0.0);

}
