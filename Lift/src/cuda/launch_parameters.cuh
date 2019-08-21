#pragma once
#include <optix.h>
#include "glm/glm.hpp"
using namespace glm;

namespace lift {
	struct TriangleMeshSbtData {
		vec3 color;
		vec3* vertices;
		ivec3* indices;
	};

	struct LaunchParameters {
		struct {
			uint32_t* color_buffer;
			ivec2 size;
		} frame;

		struct {
			vec3 position;
			vec3 direction;
			vec3 horizontal;
			vec3 vertical;
		} camera;

		OptixTraversableHandle traversable;
	};
}
