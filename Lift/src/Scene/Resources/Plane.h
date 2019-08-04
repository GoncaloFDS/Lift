#pragma once
#include "Mesh.h"
#include "pch.h"

namespace lift {

	class Plane : public Mesh {
	public:
		Plane(const int tess_u, const int tess_v, const int up_axis) {

			if (tess_u < 1 && tess_v < 1)
				LF_CORE_WARN("Plane vectors < 1");

			const float u_tile = 2.0f / float(tess_u);
			const float v_tile = 2.0f / float(tess_v);

			optix::float3 corner;

			std::vector<VertexAttributes> vertices;
			VertexAttributes vertex{};

			switch (up_axis) {
			case 0:
				corner = optix::make_float3(0.0f, -1.0f, 1.0f);
				// Lower front corner of the plane. tex_coords (0.0f, 0.0f).

				vertex.tangent = optix::make_float3(0.0f, 0.0f, -1.0f);
				vertex.normal = optix::make_float3(1.0f, 0.0f, 0.0f);

				for (int j = 0; j <= tess_v; ++j) {
					const float v = float(j) * v_tile;

					for (int i = 0; i <= tess_u; ++i) {
						const float u = float(i) * u_tile;

						vertex.vertex = corner + optix::make_float3(0.0f, v, -u);
						vertex.tex_coords = optix::make_float3(u * 0.5f, v * 0.5f, 0.0f);

						vertices.push_back(vertex);
					}
				}
				break;

			case 1: // Positive y-axis is the geometry normal, create geometry on the xz-plane.
				corner = optix::make_float3(-1.0f, 0.0f, 1.0f);
				// left front corner of the plane. tex_coords (0.0f, 0.0f).

				vertex.tangent = optix::make_float3(1.0f, 0.0f, 0.0f);
				vertex.normal = optix::make_float3(0.0f, 1.0f, 0.0f);

				for (int j = 0; j <= tess_v; ++j) {
					const float v = float(j) * v_tile;

					for (int i = 0; i <= tess_u; ++i) {
						const float u = float(i) * u_tile;

						vertex.vertex = corner + optix::make_float3(u, 0.0f, -v);
						vertex.tex_coords = optix::make_float3(u * 0.5f, v * 0.5f, 0.0f);

						vertices.push_back(vertex);
					}
				}
				break;

			case 2: // Positive z-axis is the geometry normal, create geometry on the xy-plane.
				corner = optix::make_float3(-1.0f, -1.0f, 0.0f);
				// Lower left corner of the plane. tex_coords (0.0f, 0.0f).

				vertex.tangent = optix::make_float3(1.0f, 0.0f, 0.0f);
				vertex.normal = optix::make_float3(0.0f, 0.0f, 1.0f);

				for (int j = 0; j <= tess_v; ++j) {
					const float v = float(j) * v_tile;

					for (int i = 0; i <= tess_u; ++i) {
						const float u = float(i) * u_tile;

						vertex.vertex = corner + optix::make_float3(u, v, 0.0f);
						vertex.tex_coords = optix::make_float3(u * 0.5f, v * 0.5f, 0.0f);

						vertices.push_back(vertex);
					}
				}
				break;
			default:
				LF_CORE_WARN("Invalid plane normal");
			}

			std::vector<unsigned int> indices;

			const unsigned int stride = tess_u + 1;
			for (int j = 0; j < tess_v; ++j) {
				for (int i = 0; i < tess_u; ++i) {
					indices.push_back(j * stride + i);
					indices.push_back(j * stride + i + 1);
					indices.push_back((j + 1) * stride + i + 1);
					indices.push_back((j + 1) * stride + i + 1);
					indices.push_back((j + 1) * stride + i);
					indices.push_back(j * stride + i);
				}
			}

			LF_CORE_TRACE("Created a plane: Up Axis = {0}, Vertices = {1}, Triagles = {2}", up_axis, vertices.size(),
						  indices.size() / 3);

			geometry_ = CreateGeometry(vertices, indices);
		}
	};
}
