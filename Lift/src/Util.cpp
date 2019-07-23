#include "pch.h"
#include "Util.h"
#include "Cuda/vertex_attributes.cuh"
#include "Application.h"

std::string lift::Util::GetPtxString(const char* file_name) {
	std::string ptx_source;

	const std::ifstream file(file_name);
	if (file.good()) {
		std::stringstream source_buffer;
		source_buffer << file.rdbuf();
		return source_buffer.str();
	}
	LF_CORE_ERROR("Invalid PTX path: {0}", file_name);
	return "Invalid PTX";
}

optix::Geometry lift::Util::CreateGeometry(const std::vector<VertexAttributes>& attributes,
										   const std::vector<unsigned>& indices) {
	auto& app = Application::Get();
	auto handle = app.GetOptixContext();

	optix::Geometry geometry(nullptr);

	geometry = handle->createGeometry();

	optix::Buffer attributes_buffer = handle->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
	attributes_buffer->setElementSize(sizeof(VertexAttributes));
	attributes_buffer->setSize(attributes.size());

	auto dst = attributes_buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(dst, attributes.data(), sizeof(VertexAttributes) * attributes.size());
	attributes_buffer->unmap();

	optix::Buffer indices_buffer = handle->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, indices.size() / 3);

	dst = indices_buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(dst, indices.data(), sizeof(optix::uint3) * indices.size() / 3);
	indices_buffer->unmap();

	geometry->setBoundingBoxProgram(app.GetOptixProgram("triangle_bounding_box"));

	geometry->setIntersectionProgram(app.GetOptixProgram("triangle_intersection"));

	geometry["attributes_buffer"]->setBuffer(attributes_buffer);
	geometry["indices_buffer"]->setBuffer(indices_buffer);
	geometry->setPrimitiveCount(static_cast<unsigned int>(indices.size()) / 3);

	return geometry;
}

optix::Geometry lift::Util::CreatePlane(int tess_u, int tess_v, int up_axis) {
	if (tess_u < 1 && tess_v < 1)
		LF_CORE_WARN("Plane vectors < 1");

	const float u_tile = 2.0f / float(tess_u);
	const float v_tile = 2.0f / float(tess_v);

	optix::float3 corner;

	std::vector<VertexAttributes> vertices;
	VertexAttributes vertex;

	switch (up_axis) {
	case 0:
		corner = optix::make_float3(0.0f, -1.0f, 1.0f); // Lower front corner of the plane. tex_coords (0.0f, 0.0f).

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
		corner = optix::make_float3(-1.0f, 0.0f, 1.0f); // left front corner of the plane. tex_coords (0.0f, 0.0f).

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
		corner = optix::make_float3(-1.0f, -1.0f, 0.0f); // Lower left corner of the plane. tex_coords (0.0f, 0.0f).

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

	LF_CORE_TRACE("Created plane, Up Axis {0}, Vertices {1}, Triagles {2}", up_axis, vertices.size(),
				  indices.size() / 3);

	return CreateGeometry(vertices, indices);
}
