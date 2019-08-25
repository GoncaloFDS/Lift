#include "pch.h"
#include "Mesh.h"
#include "core/Profiler.h"

lift::Mesh::Mesh(const std::string& path) : transform_(1) {
	const auto name = path.substr(path.find_last_of('/') + 1, path.back());
	Profiler profiler(std::string("Loaded mesh: ") + name);
	Assimp::Importer importer;
	const auto* scene = importer.ReadFile(
		path, aiProcess_CalcTangentSpace | aiProcess_FlipUVs | aiProcess_Triangulate);
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		LF_CORE_FATAL("Failed to load mesh \"{0}\"", name);
		LF_CORE_FATAL("assimp::error: {0}", importer.GetErrorString());
		return;
	}

	ProcessNode(scene->mRootNode, scene);
}

void lift::Mesh::ProcessNode(aiNode* node, const aiScene* scene) {
	for (unsigned mesh_id = 0; mesh_id < node->mNumMeshes; mesh_id++) {
		auto* mesh = scene->mMeshes[node->mMeshes[mesh_id]];
		//TODO deal with multiple meshes
		vertices.reserve(mesh->mNumVertices);
		indices.reserve(mesh->mNumFaces);
		for (unsigned id = 0; id < mesh->mNumVertices; id++) {
			vertices.emplace_back(mesh->mVertices[id].x, mesh->mVertices[id].y, mesh->mVertices[id].z);
		}
		for (unsigned id = 0; id < mesh->mNumFaces; id++) {
			indices.emplace_back(mesh->mFaces[id].mIndices[0], mesh->mFaces[id].mIndices[1], mesh->mFaces[id].mIndices[2]);
		}

	}

	for (unsigned i = 0; i < node->mNumChildren; i++) {
		ProcessNode(node->mChildren[i], scene);
	}
}
