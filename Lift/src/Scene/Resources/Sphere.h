#pragma once
#include "Mesh.h"

namespace lift {
	class Sphere : public Mesh {
	public:
		Sphere(const int tess_u, const int tess_v, const float radius, const float max_theta) {

			LF_ASSERT(tess_u >= 3 && tess_v >= 3, "Sphere tessalation too low");

			std::vector<VertexAttributes> attributes;
			attributes.reserve((tess_u + 1) * tess_v);

			std::vector<unsigned int> indices;
			indices.reserve(6 * tess_u * (tess_v) - 1);

			const auto phi_step = 2.0f * M_PIf / float(tess_u);
			const auto theta_step = max_theta / float(tess_v - 1);

			// Latitudinal rings.
			// Starting at the south pole going upwards on the y-axis.
			for (int latitude = 0; latitude < tess_v; latitude++) // theta angle
			{
				const auto theta = float(latitude) * theta_step;
				const auto sin_theta = sinf(theta);
				const auto cos_theta = cosf(theta);

				const auto tex_v = float(latitude) / static_cast<float>(tess_v - 1); // Range [0.0f, 1.0f]

				for (int longitude = 0; longitude <= tess_u; longitude++) // phi angle
				{
					const auto phi = float(longitude) * phi_step;
					const auto sin_phi = sinf(phi);
					const auto cos_phi = cosf(phi);

					const auto tex_u = longitude / tess_u; // Range [0.0f, 1.0f]

					// Unit sphere coordinates are the normals.
					optix::float3 normal = optix::make_float3(cos_phi * sin_theta,
															  -cos_theta, // -y to start at the south pole.
															  -sin_phi * sin_theta);
					VertexAttributes vertex_attributes{
						normal * radius,
						optix::make_float3(-sin_phi, 0.0f, -cos_phi),
						normal,
						optix::make_float3(float(tex_u), tex_v, 0.0f)
					};
					attributes.push_back(vertex_attributes);
				}
			}

			// We have generated tess_u + 1 vertices per latitude.
			const unsigned int columns = tess_u + 1;

			// Calculate indices.
			for (int latitude = 0; latitude < tess_v - 1; latitude++) {
				for (int longitude = 0; longitude < tess_u; longitude++) {
					indices.push_back(latitude * columns + longitude); // lower left
					indices.push_back(latitude * columns + longitude + 1); // lower right
					indices.push_back((latitude + 1) * columns + longitude + 1); // upper right 

					indices.push_back((latitude + 1) * columns + longitude + 1); // upper right 
					indices.push_back((latitude + 1) * columns + longitude); // upper left
					indices.push_back(latitude * columns + longitude); // lower left
				}
			}

			LF_CORE_TRACE("Created a Sphere: Vertices = {0}, Triangles = {1}", attributes.size(), indices.size() / 3);

			geometry_ = CreateGeometry(attributes, indices);
		}
	};
}
