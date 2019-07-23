#pragma once
#include <optix_world.h>
#include "Cuda/vertex_attributes.cuh"
#include <iosfwd>
#include <vector>

namespace lift {

	class Util {
	public:
		static std::string GetPtxString(const char* file_name);
		static optix::Geometry CreateGeometry(const std::vector<VertexAttributes>& attributes,
											  const std::vector<unsigned>& indices);
		static optix::Geometry CreatePlane(int tess_u, int tess_v, int up_axis);

	};
}
