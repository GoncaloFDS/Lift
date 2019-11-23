#pragma once
#include <optix.h>
#include "geometry_data.h"
#include "material_data.h"
#include "light.h"

namespace lift {

enum RayType {
    RADIANCE_RAY_TYPE = 0,
    SHADOW_RAY_TYPE = 1,
    RAY_TYPE_COUNT = 2,
};

struct HitGroupData {
    GeometryData geometry_data;
    MaterialData material_data;
};

struct LaunchParameters {
    uint32_t subframe_index{};
    float4* accum_buffer{};
    uchar4* frame_buffer{};
    int32_t max_depth{};
    uint32_t samples_per_launch{};

    struct {
        float3 eye;
        float3 u;
        float3 v;
        float3 w;
    } camera{};

    BufferView<Light> lights;
    OptixTraversableHandle handle{};

};

struct PayloadRadiance {
	uint32_t random_seed;
	float3 pixel_color;
};

struct MissData {
    float4 bg_color{};
};

struct RayGenData {

};

}
