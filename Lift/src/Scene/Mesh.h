#pragma once

#include "Aabb.h"
#include <optix.h>
#include "renderer/BufferView.h"

namespace lift {
	struct Mesh {
		std::string name;
		mat4 transform;

		std::vector<BufferView<glm::ivec3>> indices;
		std::vector<BufferView<glm::vec3>> positions;
		std::vector<BufferView<glm::vec3>> normals;
		std::vector<BufferView<glm::vec2>> tex_coords;

		std::vector<int32_t> material_idx;

		OptixTraversableHandle gas_handle = 0;
		CUdeviceptr d_gas_output = 0;

		Aabb object_aabb;
		Aabb world_aabb;
	};
}
