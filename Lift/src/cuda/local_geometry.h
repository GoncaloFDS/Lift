#pragma once

#include "optix.h"

#include "vec_math.h"
#include "renderer/BufferView.h"
#include "scene/GeometryData.h"

namespace lift {

struct LocalGeometry {
	float3 P;
	float3 N;
	float3 Ng;
	float2 UV;
	float3 dndu;
	float3 dndv;
	float3 dpdu;
	float3 dpdv;
};

__host__ __device__ LocalGeometry getLocalGeometry(const GeometryData &geometry_data) {
	LocalGeometry geometry;
	switch (geometry_data.type) {
		case GeometryData::TRIANGLE_MESH: {
			const GeometryData::TriangleMesh &mesh_data = geometry_data.triangle_mesh;

			const uint32_t prim_idx = optixGetPrimitiveIndex();
			const float2 barys = optixGetTriangleBarycentrics();

			uint3 tri = make_uint3(0u, 0u, 0u);
			if (mesh_data.indices.elmt_byte_size == 4) {
				const uint3 *indices = reinterpret_cast<uint3 *>( mesh_data.indices.data );
				tri = indices[prim_idx];
			} else {
				const uint16_t *indices = reinterpret_cast<uint16_t *>( mesh_data.indices.data );
				const uint16_t idx0 = indices[prim_idx * 3 + 0];
				const uint16_t idx1 = indices[prim_idx * 3 + 1];
				const uint16_t idx2 = indices[prim_idx * 3 + 2];
				tri = make_uint3(idx0, idx1, idx2);
			}

			const float3 P0 = mesh_data.positions[tri.x];
			const float3 P1 = mesh_data.positions[tri.y];
			const float3 P2 = mesh_data.positions[tri.z];
			geometry.P = (1.0f - barys.x - barys.y) * P0 + barys.x * P1 + barys.y * P2;
			geometry.P = optixTransformPointFromObjectToWorldSpace(geometry.P);

			float2 UV0, UV1, UV2;
			if (mesh_data.tex_coords) {
				UV0 = mesh_data.tex_coords[tri.x];
				UV1 = mesh_data.tex_coords[tri.y];
				UV2 = mesh_data.tex_coords[tri.z];
				geometry.UV = (1.0f - barys.x - barys.y) * UV0 + barys.x * UV1 + barys.y * UV2;
			} else {
				UV0 = make_float2(0.0f, 0.0f);
				UV1 = make_float2(0.0f, 1.0f);
				UV2 = make_float2(1.0f, 0.0f);
				geometry.UV = barys;
			}

			geometry.Ng = normalize(cross(P1 - P0, P2 - P0));
			geometry.Ng = optixTransformNormalFromObjectToWorldSpace(geometry.Ng);

			float3 N0, N1, N2;
			if (mesh_data.normals) {
				N0 = mesh_data.normals[tri.x];
				N1 = mesh_data.normals[tri.y];
				N2 = mesh_data.normals[tri.z];
				geometry.N = (1.0f - barys.x - barys.y) * N0 + barys.x * N1 + barys.y * N2;
				geometry.N = normalize(optixTransformNormalFromObjectToWorldSpace(geometry.N));
			} else {
				geometry.N = N0 = N1 = N2 = geometry.Ng;
			}

			const float du1 = UV0.x - UV2.x;
			const float du2 = UV1.x - UV2.x;
			const float dv1 = UV0.y - UV2.y;
			const float dv2 = UV1.y - UV2.y;

			const float3 dp1 = P0 - P2;
			const float3 dp2 = P1 - P2;

			const float3 dn1 = N0 - N2;
			const float3 dn2 = N1 - N2;

			const float det = du1 * dv2 - dv1 * du2;

			const float invdet = 1.f / det;
			geometry.dpdu = (dv2 * dp1 - dv1 * dp2) * invdet;
			geometry.dpdv = (-du2 * dp1 + du1 * dp2) * invdet;
			geometry.dndu = (dv2 * dn1 - dv1 * dn2) * invdet;
			geometry.dndu = (-du2 * dn1 + du1 * dn2) * invdet;

			break;
		}
		case GeometryData::SPHERE: {
			break;
		}
		default: break;
	}

	return geometry;
}
}