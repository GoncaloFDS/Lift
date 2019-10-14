#include <optix_device.h>

#include "launch_parameters.h"
#include "vector_functions.hpp"
#include "random.cuh"
#include "vec_math.h"
#include "local_geometry.h"

using namespace lift;
extern "C" __constant__ LaunchParameters params;

//------------------------------------------------------------------------------
//
// GGX/smith shading helpers
// TODO: move into header so can be shared by path tracer and bespoke renderers
//
//------------------------------------------------------------------------------

__device__ float3 schlick(const float3 spec_color, const float v_dot_h) {
    return spec_color + (makeFloat3(1.0f) - spec_color) * powf(1.0f - v_dot_h, 5.0f);
}

__device__ float vis(const float n_dot_l, const float n_dot_v, const float alpha) {
    const float alpha_sq = alpha * alpha;

    const float ggx_0 = n_dot_l * sqrtf(n_dot_v * n_dot_v * (1.0f - alpha_sq) + alpha_sq);
    const float ggx_1 = n_dot_v * sqrtf(n_dot_l * n_dot_l * (1.0f - alpha_sq) + alpha_sq);

    return 2.0f * n_dot_l * n_dot_v / (ggx_0 + ggx_1);
}

__device__ float ggxNormal(const float n_dot_h, const float alpha) {
    const float alpha_sq = alpha * alpha;
    const float n_dot_h_sq = n_dot_h * n_dot_h;
    const float x = n_dot_h_sq * (alpha_sq - 1.0f) + 1.0f;
    return alpha_sq / (M_PIf * x * x);
}

__device__ float3 linearize(float3 c) {
    return make_float3(
        powf(c.x, 2.2f),
        powf(c.y, 2.2f),
        powf(c.z, 2.2f)
    );
}

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float tmin,
    float tmax,
    PayloadRadiance* payload
) {
    uint32_t u_0 = 0, u_1 = 0, u_2 = 0, u_3 = 0;
    optixTrace(
        handle,
        ray_origin, ray_direction,
        tmin,
        tmax,
        0.0f,                     // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,        // SBT offset
        RAY_TYPE_COUNT,           // SBT stride
        RAY_TYPE_RADIANCE,        // missSBTIndex
        u_0, u_1, u_2, u_3);

    payload->result.x = __int_as_float(u_0);
    payload->result.y = __int_as_float(u_1);
    payload->result.z = __int_as_float(u_2);
    payload->depth = u_3;
}

static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float tmin,
    float tmax
) {
    uint32_t occluded = 0u;
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                    // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        RAY_TYPE_OCCLUSION,      // SBT offset
        RAY_TYPE_COUNT,          // SBT stride
        RAY_TYPE_OCCLUSION,      // missSBTIndex
        occluded);
    return occluded;
}
__forceinline__ __device__ void setPayloadResult(float3 p) {
    optixSetPayload_0(float_as_int(p.x));
    optixSetPayload_1(float_as_int(p.y));
    optixSetPayload_2(float_as_int(p.z));
}

__forceinline__ __device__ void setPayloadOcclusion(bool occluded) {
    optixSetPayload_0(static_cast<uint32_t>( occluded ));
}

extern "C" __global__ void __closesthit__radiance() {
    const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry_data);

    ////
    // Retrieve material data
    //
    float3 base_color = makeFloat3(hit_group_data->material_data.base_color);
    if (hit_group_data->material_data.base_color_tex)
        base_color *= linearize(makeFloat3(
            tex2D<float4>(hit_group_data->material_data.base_color_tex, geom.uv.x, geom.uv.y)
        ));

    float metallic = hit_group_data->material_data.metallic;
    float roughness = hit_group_data->material_data.roughness;
    float4 mr_tex = makeFloat4(1.0f);
    if (hit_group_data->material_data.metallic_roughness_tex)
        // MR tex is (occlusion, roughness, metallic )
        mr_tex = tex2D<float4>(hit_group_data->material_data.metallic_roughness_tex, geom.uv.x, geom.uv.y);
    roughness *= mr_tex.y;
    metallic *= mr_tex.z;


    //
    // Convert to material params
    //
    const float f_0 = 0.04f;
    const float3 diff_color = base_color * (1.0f - f_0) * (1.0f - metallic);
    const float3 spec_color = lerp(makeFloat3(f_0), base_color, metallic);
    const float alpha = roughness * roughness;

    //
    // compute direct lighting
    //

    float3 n = geom.n;
    if (hit_group_data->material_data.normal_tex) {
        const float4 nn = 2.0f * tex2D<float4>(hit_group_data->material_data.normal_tex, geom.uv.x, geom.uv.y)
            - makeFloat4(1.0f);
        n = normalize(nn.x * normalize(geom.dpdu) + nn.y * normalize(geom.dpdv) + nn.z * geom.n);
    }

    float3 result = makeFloat3(0.0f);

    for (int i = 0; i < params.lights.count; ++i) {
        Light::Point light = params.lights[i];

        // TODO: optimize
        const float l_dist = length(light.position - geom.p);
        const float3 l = (light.position - geom.p) / l_dist;
        const float3 v = -normalize(optixGetWorldRayDirection());
        const float3 h = normalize(l + v);
        const float n_dot_l = dot(n, l);
        const float n_dot_v = dot(n, v);
        const float n_dot_h = dot(n, h);
        const float v_dot_h = dot(v, h);

        if (n_dot_l > 0.0f && n_dot_v > 0.0f) {
            const float tmin = 0.001f;          // TODO
            const float tmax = l_dist - 0.001f; // TODO
            const bool occluded = traceOcclusion(params.handle, geom.p, l, tmin, tmax);
            if (!occluded) {
                const float3 f = schlick(spec_color, v_dot_h);
                const float g_vis = vis(n_dot_l, n_dot_v, alpha);
                const float d = ggxNormal(n_dot_h, alpha);

                const float3 diff = (1.0f - f) * diff_color / M_PIf;
                const float3 spec = f * g_vis * d;

                result += light.color * light.intensity * n_dot_l * (diff + spec);
            }
        }
    }
    // TODO: add debug viewing mode that allows runtime switchable views of shading params, normals, etc
    //result = make_float3( roughness );
    //result = N*0.5f + make_float3( 0.5f );
    //result = geom.N*0.5f + make_float3( 0.5f );
    setPayloadResult(result);
}

extern "C" __global__ void __miss__radiance() {
    setPayloadResult(params.miss_color);
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__render_frame() {
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint3 launch_dims = optixGetLaunchDimensions();
    const float3 eye = params.camera.eye;
    const float3 u = params.camera.u;
    const float3 v = params.camera.v;
    const float3 w = params.camera.w;
    const int subframe_index = params.subframe_index;

    //
    // Generate camera ray
    //
    uint32_t seed = tea<4>(launch_idx.y * launch_dims.x + launch_idx.x, subframe_index);
    const float2 subpixel_jitter = subframe_index == 0 ?
                                   make_float2(0.0f, 0.0f) :
                                   make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

    const float2 d = 2.0f * make_float2(
        (static_cast<float>( launch_idx.x ) + subpixel_jitter.x) / static_cast<float>( launch_dims.x ),
        (static_cast<float>( launch_idx.y ) + subpixel_jitter.y) / static_cast<float>( launch_dims.y )
    ) - 1.0f;
    const float3 ray_direction = normalize(d.x * u + d.y * v + w);
    const float3 ray_origin = eye;

    //
    // Trace camera ray
    //
    PayloadRadiance payload{
        makeFloat3(0.0f),
        1.0f,
        0
    };

    traceRadiance(
        params.handle,
        ray_origin,
        ray_direction,
        0.01f,
        1e16f,
        &payload
    );

    //
    // Update results
    //
    const uint32_t image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    float3 accum_color = payload.result;

    if (subframe_index > 0) {
        const float a = 1.0f / static_cast<float>( subframe_index + 1 );
        const float3 accum_color_prev = makeFloat3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = makeFloat4(accum_color, 1.0f);
    params.frame_buffer[image_index] = makeColor(accum_color);

}
