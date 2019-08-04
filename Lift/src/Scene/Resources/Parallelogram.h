#pragma once
#include "Mesh.h"
#include <pch.h>

namespace lift {

	class Parallelogram : public lift::Mesh {
	public:
		Parallelogram(const optix::float3& position, const optix::float3& vector_u,
					  const optix::float3& vector_v, const optix::float3& normal) : Mesh() {

			std::vector<VertexAttributes> attributes;

			VertexAttributes attrib;

			// Same for all four vertices in this parallelogram.
			attrib.tangent = optix::normalize(vector_u);
			attrib.normal = normal;

			attrib.vertex = position; // left bottom
			attrib.tex_coords = optix::make_float3(0.0f, 0.0f, 0.0f);
			attributes.push_back(attrib);

			attrib.vertex = position + vector_u; // right bottom
			attrib.tex_coords = optix::make_float3(1.0f, 0.0f, 0.0f);
			attributes.push_back(attrib);

			attrib.vertex = position + vector_u + vector_v; // right top
			attrib.tex_coords = optix::make_float3(1.0f, 1.0f, 0.0f);
			attributes.push_back(attrib);

			attrib.vertex = position + vector_v; // left top
			attrib.tex_coords = optix::make_float3(0.0f, 1.0f, 0.0f);
			attributes.push_back(attrib);

			std::vector<unsigned int> indices;

			indices.push_back(0);
			indices.push_back(1);
			indices.push_back(2);

			indices.push_back(2);
			indices.push_back(3);
			indices.push_back(0);

			LF_CORE_TRACE("Created a Parallelogram: Vertices = {0}, Triangles = {1}", attributes.size(), indices.size() / 3);

			geometry_ = CreateGeometry(attributes, indices);

		}

	};
}
