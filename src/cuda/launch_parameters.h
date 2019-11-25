#pragma once
#include <optix.h>
#include "geometry_data.h"
#include "material_data.h"
#include "light.h"
#include "random.cuh"

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
    uint32_t width{};
    uint32_t height{};
	uint32_t samples_per_launch{};
	uint32_t samples_per_light{};

	struct {
        float3 eye;
        float3 u;
        float3 v;
        float3 w;
    } camera{};

    BufferView<Light> lights;
    OptixTraversableHandle handle{};

};

struct RadiancePRD {
	Random random;
	float3 emitted;
	float3 radiance;
	float3 attenuation;
	float3 origin;
	float3 direction;
	int32_t count_emitted;
	int32_t done;
	int32_t pad;
};

struct MissData {
    float4 bg_color{};
};

struct RayGenData {

};

}
