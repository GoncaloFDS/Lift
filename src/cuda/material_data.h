#pragma once

#include <cuda_runtime.h>

namespace lift {

struct MaterialData {
    float4 base_color = {1.0f, 1.0f, 1.0f, 1.0f};
	float3 emission_color = {0.0f, 0.0f, 0.0f};
	float metallic = 1.0f;
    float roughness = 1.0f;

    cudaTextureObject_t base_color_tex = 0;
    cudaTextureObject_t metallic_roughness_tex = 0;
    cudaTextureObject_t normal_tex = 0;
};
}
