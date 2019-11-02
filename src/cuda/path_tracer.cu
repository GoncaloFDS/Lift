#include <optix.h>

#include "launch_parameters.h"
#include "vector_functions.hpp"
#include "random.cuh"
#include "vec_math.h"
#include "local_geometry.h"
#include "cuda_util.cuh"

using namespace lift;
extern "C" __constant__ LaunchParameters params;

__forceinline__ __device__ void setPayloadResult(float3 p) {
    optixSetPayload_0(float_as_int(p.x));
    optixSetPayload_1(float_as_int(p.y));
    optixSetPayload_2(float_as_int(p.z));
}

static __forceinline__ __device__ PayloadRadiance* getPayload() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<PayloadRadiance*>( unpackPointer(u0, u1));
}

struct Onb {
    __forceinline__ __device__ Onb(const float3 normal) {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z)) {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        } else {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }
    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float t_min,
    float t_max,
    PayloadRadiance* payload) {

    uint32_t u_0, u_1;
    packPointer(payload, u_0, u_1);
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        t_min,
        t_max,
        0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,        // SBT offset
        RAY_TYPE_COUNT,           // SBT stride
        RAY_TYPE_RADIANCE,        // missSBTIndex
        u_0, u_1);
}

static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float t_min,
    float t_max) {
    
    uint32_t occluded = 0u;
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        t_min,
        t_max,
        0.0f,                    // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        RAY_TYPE_OCCLUSION,      // SBT offset
        RAY_TYPE_COUNT,          // SBT stride
        RAY_TYPE_OCCLUSION,      // missSBTIndex
        occluded);
    return occluded;
}

extern "C" __global__ void __raygen__render_frame() {
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint3 launch_dims = optixGetLaunchDimensions();
    const float3 eye = params.camera.eye;
    const float3 U = params.camera.u;
    const float3 V = params.camera.v;
    const float3 W = params.camera.w;
    const int subframe_index = params.subframe_index;
    
    uint32_t seed = tea<4>(launch_idx.y * launch_dims.x + launch_idx.x, subframe_index);
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    int i = params.samples_per_launch;
    do {
        const float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.0f);

        const float2 d = 2.0f * make_float2(
            (static_cast<float>( launch_idx.x ) + subpixel_jitter.x) / static_cast<float>( launch_dims.x ),
            (static_cast<float>( launch_idx.y ) + subpixel_jitter.y) / static_cast<float>( launch_dims.y )
        ) - 1.0f;
        float3 ray_direction = normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        PayloadRadiance payload{};
        payload.emitted = make_float3(0.f, 0.0f, 0.0f);
        payload.radiance = make_float3(0.f, 0.0f, 0.0f);
        payload.attenuation = make_float3(1.f, 1.0f, 1.0f);
        payload.countEmitted = true;
        payload.done = false;
        payload.seed = seed;

        int depth = 0;
        for (;;) {
            traceRadiance(
                params.handle,
                ray_origin,
                ray_direction,
                0.01f,  // tmin       // TODO: smarter offset
                1e16f,  // tmax
                &payload);

            result += payload.emitted;
            result += payload.radiance * payload.attenuation;

            if (payload.done || depth >= 3) // TODO RR, variable for depth
                break;

            ray_origin = payload.origin;
            ray_direction = payload.direction;

            ++depth;
        }
    } while (--i);

    const uint32_t image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    float3 accum_color = result / static_cast<float>( params.samples_per_launch );

    if (subframe_index > 0) {
        const float a = 1.0f / static_cast<float>( subframe_index + 1 );
        const float3 accum_color_prev = makeFloat3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = makeFloat4(accum_color, 1.0f);
    params.frame_buffer[image_index] = makeColor(accum_color);

}

extern "C" __global__ void __miss__radiance() {
    MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    PayloadRadiance* payload = getPayload();
    
    payload->radiance = makeFloat3(rt_data->bg_color);
    payload->done = true;
}

extern "C" __global__ void __closesthit__occlusion() {
    setPayloadOcclusion(true);
}

extern "C" __global__ void __closesthit__radiance() {
    const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry_data);

    PayloadRadiance* payload = getPayload();
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
        auto light = params.lights[i];

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
    //setPayloadResult(result);
    payload->radiance = result;
}















