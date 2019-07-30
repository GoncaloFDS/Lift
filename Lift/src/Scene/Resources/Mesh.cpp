#include "pch.h"
#include "Mesh.h"
#include "Core/Profiler.h"
#include "Application.h"
#include "Platform/Optix/OptixContext.h"

lift::Mesh::Mesh(const std::string& path) : transform_(1) {
	const auto name = path.substr(path.find_last_of('/') + 1, path.back());
	Profiler profiler(std::string("Loaded mesh: ") + name);
	Assimp::Importer importer;
	const auto* scene = importer.ReadFile(path, aiProcess_CalcTangentSpace | aiProcess_FlipUVs | aiProcess_Triangulate);
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		LF_CORE_FATAL("Failed to load mesh \"{0}\"", name);
		LF_CORE_FATAL("assimp::error: {0}", importer.GetErrorString());
		return;
	}

	ProcessNode(scene->mRootNode, scene);

	geometry_ = CreateGeometry(mesh_data_.attributes, mesh_data_.indices);
}

lift::Mesh::Mesh(const Geometry geometry) : transform_(1) {
	switch (geometry) {
	case Geometry::Plane:
		geometry_ = CreatePlaneGeometry(1, 1, 1);
		break;
	case Geometry::Sphere:
		geometry_ = CreateSphereGeometry(180, 90, 1.0f, M_PIf/2);
		break;
	default: ;
	}
}

void lift::Mesh::ProcessNode(aiNode* node, const aiScene* scene) {
	for (unsigned i = 0; i < node->mNumMeshes; i++) {
		auto* mesh = scene->mMeshes[node->mMeshes[i]];
		mesh_data_ = TranslateMesh(mesh, scene);
		//TODO deal with multiple meshes
	}

	for (unsigned i = 0; i < node->mNumChildren; i++) {
		ProcessNode(node->mChildren[i], scene);
	}
}

MeshData lift::Mesh::TranslateMesh(aiMesh* mesh, const aiScene* scene) const {
	std::vector<VertexAttributes> vertex_attributes;
	std::vector<unsigned> indices;

	for (unsigned i = 0; i < mesh->mNumVertices; i++) {
		vertex_attributes.push_back({
			optix::make_float3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z),
			optix::make_float3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z),
			optix::make_float3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z),
			optix::make_float3(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y, 0.0f)
		});
	}
	for (unsigned i = 0; i < mesh->mNumFaces; i++) {
		const auto face = mesh->mFaces[i];
		indices.push_back(face.mIndices[0]);
		indices.push_back(face.mIndices[1]);
		indices.push_back(face.mIndices[2]);
	}
	return {vertex_attributes, indices};
}

void lift::Mesh::SubmitMesh(optix::Group& group) {
	auto& optix_context = OptixContext::Get();
	auto geometry_instance = optix_context->createGeometryInstance();
	geometry_instance->setGeometry(geometry_);
	geometry_instance->setMaterialCount(1);
	geometry_instance->setMaterial(0, material_);
	geometry_instance["per_material_index"]->setInt(0);

	auto acceleration = optix_context->createAcceleration("Trbvh");
	acceleration->setProperty("vertex_buffer_name", "attributes_buffer");
	acceleration->setProperty("vertex_buffer_stride", "48");
	acceleration->setProperty("indices_buffer_name", "indices_buffer");
	acceleration->setProperty("indices_buffer_stride", "12");

	auto geometry_group = optix_context->createGeometryGroup();
	geometry_group->setAcceleration(acceleration);
	geometry_group->setChildCount(1);
	geometry_group->setChild(0, geometry_instance);

	auto optix_transform = optix_context->createTransform();
	optix_transform->setChild(geometry_group);
	optix_transform->setMatrix(true, value_ptr(transform_), value_ptr(inverse(transform_)));

	const auto count = group->getChildCount();
	group->setChildCount(count + 1);
	group->setChild(count, optix_transform);
}

optix::Geometry lift::Mesh::CreatePlaneGeometry(const int tess_u, const int tess_v, const int up_axis) const {
	if (tess_u < 1 && tess_v < 1)
		LF_CORE_WARN("Plane vectors < 1");

	const float u_tile = 2.0f / float(tess_u);
	const float v_tile = 2.0f / float(tess_v);

	optix::float3 corner;

	std::vector<VertexAttributes> vertices;
	VertexAttributes vertex{};

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

	LF_CORE_TRACE("Created a plane: Up Axis = {0}, Vertices = {1}, Triagles = {2}", up_axis, vertices.size(),
				  indices.size() / 3);

	return CreateGeometry(vertices, indices);
}

optix::Geometry lift::Mesh::CreateSphereGeometry(const int tess_u, const int tess_v, const float radius,
												 const float max_theta) const {
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

	return CreateGeometry(attributes, indices);
}

optix::Geometry lift::Mesh::CreateGeometry(const std::vector<VertexAttributes>& attributes,
										   const std::vector<unsigned>& indices) {
	auto& app = Application::Get();
	auto& optix_context = OptixContext::Get();

	optix::Geometry geometry(nullptr);

	geometry = optix_context->createGeometry();

	optix::Buffer attributes_buffer = optix_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
	attributes_buffer->setElementSize(sizeof(VertexAttributes));
	attributes_buffer->setSize(attributes.size());

	auto dst = attributes_buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	memcpy(dst, attributes.data(), sizeof(VertexAttributes) * attributes.size());
	attributes_buffer->unmap();

	optix::Buffer indices_buffer = optix_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, indices.size() / 3);

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
