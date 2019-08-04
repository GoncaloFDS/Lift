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
	geometry_instance_ = optix_context->createGeometryInstance();
	geometry_instance_->setGeometry(geometry_);
	geometry_instance_->setMaterialCount(1);
	geometry_instance_->setMaterial(0, material_);
	geometry_instance_["per_material_index"]->setInt(0);

	auto acceleration = optix_context->createAcceleration("Trbvh");
	acceleration->setProperty("vertex_buffer_name", "attributes_buffer");
	acceleration->setProperty("vertex_buffer_stride", "48");
	acceleration->setProperty("indices_buffer_name", "indices_buffer");
	acceleration->setProperty("indices_buffer_stride", "12");

	auto geometry_group = optix_context->createGeometryGroup();
	geometry_group->setAcceleration(acceleration);
	geometry_group->setChildCount(1);
	geometry_group->setChild(0, geometry_instance_);

	auto optix_transform = optix_context->createTransform();
	optix_transform->setChild(geometry_group);
	optix_transform->setMatrix(true, value_ptr(transform_), value_ptr(inverse(transform_)));

	const auto count = group->getChildCount();
	group->setChildCount(count + 1);
	group->setChild(count, optix_transform);
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

	optix::Buffer indices_buffer = optix_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3,
															   indices.size() / 3);

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
