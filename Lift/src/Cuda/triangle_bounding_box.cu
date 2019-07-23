#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include "vertex_attributes.cuh"

rtBuffer<VertexAttributes> attributes_buffer;
rtBuffer<uint3> indices_buffer;

RT_PROGRAM void triangle_bounding_box(int primitive_index, float result[6]) {
	const uint3 indices = indices_buffer[primitive_index];

	const float3 v0 = attributes_buffer[indices.x].vertex;
	const float3 v1 = attributes_buffer[indices.y].vertex;
	const float3 v2 = attributes_buffer[indices.z].vertex;

	const float area = optix::length(optix::cross(v1 - v0, v2 - v0));

	optix::Aabb* aabb = reinterpret_cast<optix::Aabb*>(result);


	if (0.0f < area && !isinf(area)) {
		aabb->m_min = fminf(fminf(v0, v1), v2);
		aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
	}
	else {
		aabb->invalidate();
	}
}
