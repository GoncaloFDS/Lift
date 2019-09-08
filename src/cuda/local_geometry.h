#pragma once

#include "optix.h"

#include "vec_math.h"
#include "renderer/buffer_view.h"
#include "scene/geometry_data.h"

namespace lift {

struct LocalGeometry {
    float3 p;
    float3 n;
    float3 ng;
    float2 uv;
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
            if (mesh_data.indices.elmt_byte_size==4) {
                const uint3 *indices = reinterpret_cast<uint3 *>( mesh_data.indices.data );
                tri = indices[prim_idx];
            } else {
                const uint16_t *indices = reinterpret_cast<uint16_t *>( mesh_data.indices.data );
                const uint16_t idx_0 = indices[prim_idx*3 + 0];
                const uint16_t idx_1 = indices[prim_idx*3 + 1];
                const uint16_t idx_2 = indices[prim_idx*3 + 2];
                tri = make_uint3(idx_0, idx_1, idx_2);
            }

            const float3 p_0 = mesh_data.positions[tri.x];
            const float3 p_1 = mesh_data.positions[tri.y];
            const float3 p_2 = mesh_data.positions[tri.z];
            geometry.p = (1.0f - barys.x - barys.y)*p_0 + barys.x*p_1 + barys.y*p_2;
            geometry.p = optixTransformPointFromObjectToWorldSpace(geometry.p);

            float2 uv_0, uv_1, uv_2;
            if (mesh_data.tex_coords) {
                uv_0 = mesh_data.tex_coords[tri.x];
                uv_1 = mesh_data.tex_coords[tri.y];
                uv_2 = mesh_data.tex_coords[tri.z];
                geometry.uv = (1.0f - barys.x - barys.y)*uv_0 + barys.x*uv_1 + barys.y*uv_2;
            } else {
                uv_0 = make_float2(0.0f, 0.0f);
                uv_1 = make_float2(0.0f, 1.0f);
                uv_2 = make_float2(1.0f, 0.0f);
                geometry.uv = barys;
            }

            geometry.ng = normalize(cross(p_1 - p_0, p_2 - p_0));
            geometry.ng = optixTransformNormalFromObjectToWorldSpace(geometry.ng);

            float3 n_0, n_1, n_2;
            if (mesh_data.normals) {
                n_0 = mesh_data.normals[tri.x];
                n_1 = mesh_data.normals[tri.y];
                n_2 = mesh_data.normals[tri.z];
                geometry.n = (1.0f - barys.x - barys.y)*n_0 + barys.x*n_1 + barys.y*n_2;
                geometry.n = normalize(optixTransformNormalFromObjectToWorldSpace(geometry.n));
            } else {
                geometry.n = n_0 = n_1 = n_2 = geometry.ng;
            }

            const float du_1 = uv_0.x - uv_2.x;
            const float du_2 = uv_1.x - uv_2.x;
            const float dv_1 = uv_0.y - uv_2.y;
            const float dv_2 = uv_1.y - uv_2.y;

            const float3 dp_1 = p_0 - p_2;
            const float3 dp_2 = p_1 - p_2;

            const float3 dn_1 = n_0 - n_2;
            const float3 dn_2 = n_1 - n_2;

            const float det = du_1*dv_2 - dv_1*du_2;

            const float invdet = 1.f/det;
            geometry.dpdu = (dv_2*dp_1 - dv_1*dp_2)*invdet;
            geometry.dpdv = (-du_2*dp_1 + du_1*dp_2)*invdet;
            geometry.dndu = (dv_2*dn_1 - dv_1*dn_2)*invdet;
            geometry.dndu = (-du_2*dn_1 + du_1*dn_2)*invdet;

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