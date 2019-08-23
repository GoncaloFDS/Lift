#include "pch.h"
#include "Mesh.h"
#include "core/Profiler.h"

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

}

void lift::Mesh::ProcessNode(aiNode* node, const aiScene* scene) {
	for (unsigned i = 0; i < node->mNumMeshes; i++) {
		auto* mesh = scene->mMeshes[node->mMeshes[i]];
		//TODO deal with multiple meshes
		vertices.reserve(mesh->mNumVertices);
		indices.reserve(mesh->mNumFaces);
		for (int i = 0; i < mesh->mNumVertices; i++) {
			vertices.emplace_back(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
		}
		for (int i = 0; i < mesh->mNumFaces; i++) {
			indices.emplace_back(mesh->mFaces[i].mIndices[0], mesh->mFaces[i].mIndices[1], mesh->mFaces[i].mIndices[2]);
		}
		
	}

	for (unsigned i = 0; i < node->mNumChildren; i++) {
		ProcessNode(node->mChildren[i], scene);
	}
}
