#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

struct MeshData {
	std::vector<VertexAttributes> attributes;
	std::vector<unsigned> indices;
};

namespace lift {
	class Mesh {

	public:
		Mesh(const std::string& path);

		optix::Geometry CreateGeometry() const;
	private:
		MeshData mesh_data_;

		void ProcessNode(aiNode* node, const aiScene* scene);
		MeshData TranslateMesh(aiMesh* mesh, const aiScene* scene) const;


	};
}
