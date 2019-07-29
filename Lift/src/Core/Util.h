#pragma once
#include <optix_world.h>
#include "Cuda/vertex_attributes.cuh"
#include <iosfwd>
#include <vector>

namespace lift {

	class Util {
	public:
		static std::string GetPtxString(const char* file_name);
	};
}
