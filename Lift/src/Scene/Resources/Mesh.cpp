#include "pch.h"
#include "Mesh.h"
#include "Core/Profiler.h"

lift::Mesh::Mesh(const std::string& path) {
	auto name = path.substr(path.find_last_of('/') + 1, path.back());
	Profiler profiler(std::string("Loaded mesh: ") + name);
	Assimp::Importer importer;
	const auto* scene = importer.ReadFile(path, aiProcess_CalcTangentSpace | aiProcess_FlipUVs | aiProcess_Triangulate);
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		LF_CORE_FATAL("Failed to load mesh \"{0}\"", name);
		LF_CORE_FATAL("assimp::error: {0}", importer.GetErrorString());
		return;
	}

	ProcessNode(scene->mRootNode, scene);
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

optix::Geometry lift::Mesh::CreateGeometry() const {
	return Util::CreateGeometry(mesh_data_.attributes, mesh_data_.indices);
}
