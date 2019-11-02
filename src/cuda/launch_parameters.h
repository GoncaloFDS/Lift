#pragma once
#include <optix.h>
#include "geometry_data.h"
#include "material_data.h"
#include "light.h"

namespace lift {

enum RayType {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT = 2,
};

struct HitGroupData {
    GeometryData geometry_data;
    MaterialData material_data;
};

struct LaunchParameters {
    uint32_t subframe_index;
    float4* accum_buffer;
    uchar4* frame_buffer;
    int32_t max_depth;
    uint32_t samples_per_launch;

    struct {
        float3 eye;
        float3 u;
        float3 v;
        float3 w;
    } camera;

    BufferView<Lights::ParallelogramLight> lights;
    OptixTraversableHandle handle;

};

struct PayloadRadiance {
    // TODO: move some state directly into payload registers?
    float3 emitted;
    float3 radiance;
    float3 attenuation;
    float3 origin;
    float3 direction;
    uint32_t seed;
    int32_t countEmitted;
    int32_t done;
    int32_t pad;
};

struct MissData {
    float4 bg_color{};
};

struct RayGenData {

};

}
