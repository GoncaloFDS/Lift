#pragma once

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

struct MeshData {
	std::vector<VertexAttributes> attributes;
	std::vector<unsigned> indices;
};

namespace lift {
	class Mesh {
	public:
		Mesh() : transform_(1.0f){}
		Mesh(const std::string& path);

		[[nodiscard]] const optix::Material& Material() const { return material_; }
		void SetMaterial(const optix::Material& material) { material_ = material; }

		[[nodiscard]] const mat4& Transform() const { return transform_; }
		void SetTransform(const mat4& transform) { transform_ = transform; }

		void SubmitMesh(optix::Group& group);

	protected:
		MeshData mesh_data_;
		optix::Material material_;
		optix::Geometry geometry_;
		mat4 transform_;

		void ProcessNode(aiNode* node, const aiScene* scene);
		MeshData TranslateMesh(aiMesh* mesh, const aiScene* scene) const;

		static optix::Geometry CreateGeometry(const std::vector<VertexAttributes>& attributes,
											  const std::vector<unsigned>& indices);
	};
}
