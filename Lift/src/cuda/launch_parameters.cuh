#pragma once
#include "glm/glm.hpp"
using namespace glm;

namespace lift {
	struct LaunchParameters {
		int frame_id{0};
		uint32_t* color_buffer;
		ivec2 frame_buffer_size;
	};
}
