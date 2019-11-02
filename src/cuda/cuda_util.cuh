#pragma once
#include "launch_parameters.h"

HOSTDEVICE float3 schlick(const float3 spec_color, const float v_dot_h) {
    return spec_color + (makeFloat3(1.0f) - spec_color) * powf(1.0f - v_dot_h, 5.0f);
}

HOSTDEVICE float vis(const float n_dot_l, const float n_dot_v, const float alpha) {
    const float alpha_sq = alpha * alpha;

    const float ggx_0 = n_dot_l * sqrtf(n_dot_v * n_dot_v * (1.0f - alpha_sq) + alpha_sq);
    const float ggx_1 = n_dot_v * sqrtf(n_dot_l * n_dot_l * (1.0f - alpha_sq) + alpha_sq);

    return 2.0f * n_dot_l * n_dot_v / (ggx_0 + ggx_1);
}

HOSTDEVICE float ggxNormal(const float n_dot_h, const float alpha) {
    const float alpha_sq = alpha * alpha;
    const float n_dot_h_sq = n_dot_h * n_dot_h;
    const float x = n_dot_h_sq * (alpha_sq - 1.0f) + 1.0f;
    return alpha_sq / (M_PIf * x * x);
}

HOSTDEVICE float3 linearize(float3 c) {
    return make_float3( powf(c.x, 2.2f), powf(c.y, 2.2f), powf(c.z, 2.2f) );
}

static __forceinline__ __device__ void* unpackPointer(uint32_t i0, uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void* ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ void setPayloadOcclusion(bool occluded) {
    optixSetPayload_0(static_cast<uint32_t>( occluded ));
}

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p) {
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}
