#pragma once

#include "Aabb.h"
#include <optix.h>
#include "renderer/BufferView.h"
#include "vector_types.h"

namespace lift {
struct Mesh {
    std::string name;
    mat4 transform;

    std::vector<BufferView<uint32_t>> indices;
    std::vector<BufferView<float3>> positions;
    std::vector<BufferView<float3>> normals;
    std::vector<BufferView<float2>> tex_coords;

    std::vector<int32_t> material_idx;

    OptixTraversableHandle gas_handle = 0;
    CUdeviceptr d_gas_output = 0;

    Aabb object_aabb;
    Aabb world_aabb;
};
}
