#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "vertex_attributes.cuh"

rtBuffer<VertexAttributes> attributes_buffer;
rtBuffer<uint3> indices_buffer;

rtDeclareVariable(optix::float3, var_geometry_normal, attribute GEO_NORMAL, );
rtDeclareVariable(optix::float3, var_tangent, attribute TANGENT, );
rtDeclareVariable(optix::float3, var_normal, attribute NORMAL, );
rtDeclareVariable(optix::float3, var_tex_coord, attribute TEXCOORD, );

rtDeclareVariable(optix::Ray, the_ray, rtCurrentRay, );

// Intersection routine for indexed interleaved triangle data.
RT_PROGRAM void triangle_intersection(int primitive_index) {
	const uint3 indices = indices_buffer[primitive_index];

	VertexAttributes const& a0 = attributes_buffer[indices.x];
	VertexAttributes const& a1 = attributes_buffer[indices.y];
	VertexAttributes const& a2 = attributes_buffer[indices.z];

	const float3 v0 = a0.vertex;
	const float3 v1 = a1.vertex;
	const float3 v2 = a2.vertex;

	float3 n;
	float t;
	float beta;
	float gamma;

	if (optix::intersect_triangle(the_ray, v0, v1, v2, n, t, beta, gamma)) {
		if (rtPotentialIntersection(t)) {
			// Barycentric interpolation:
			const float alpha = 1.0f - beta - gamma;

			// Note: No normalization on the TBN attributes here for performance reasons.
			//       It's done after the transformation into world space anyway.
			var_geometry_normal = n;
			var_tangent = a0.tangent * alpha + a1.tangent * beta + a2.tangent * gamma;
			var_normal = a0.normal * alpha + a1.normal * beta + a2.normal * gamma;
			var_tex_coord = a0.tex_coords * alpha + a1.tex_coords * beta + a2.tex_coords * gamma;

			rtReportIntersection(0);
		}
	}
}
