#pragma once
#include <optix.h>
#include "scene/geometry_data.h"
#include "scene/material_data.h"
#include "scene/light.h"

namespace lift {

struct HitGroupData {
    GeometryData geometry_data;
    MaterialData material_data;
};

struct LaunchParameters {
    uint32_t subframe_index;
    float4* accum_buffer;
    uchar4* frame_buffer;
    int32_t max_depth;

    struct {
        float3 eye;
        float3 u;
        float3 v;
        float3 w;
    } camera;

    BufferView<Light::Point> lights;
    float3 miss_color;
    OptixTraversableHandle handle;

};

struct PayloadRadiance {
    float3 result;
    float importance;
    int depth;
};

enum RayType {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT = 2,
};

}
