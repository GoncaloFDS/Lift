#pragma once
#include <optix.h>
#include "scene/GeometryData.h"
#include "scene/MaterialData.h"
using namespace glm;

namespace lift {

	struct HitGroupData {
		GeometryData geometry_data;
		MaterialData material_data;
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
