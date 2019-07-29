#pragma once

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

struct MeshData {
	std::vector<VertexAttributes> attributes;
	std::vector<unsigned> indices;
};

enum class Geometry {
	Plane,
	Sphere
};

namespace lift {
	class Mesh {
	public:
		Mesh(const std::string& path);
		Mesh(Geometry geometry);

		[[nodiscard]] const optix::Material& Material() const { return material_; }
		void SetMaterial(const optix::Material& material) { material_ = material; }

		[[nodiscard]] const mat4& Transform() const { return transform_; }
		void SetTransform(const mat4& transform) { transform_ = transform; }

		void SubmitMesh(optix::Group& group);

	private:
		MeshData mesh_data_;
		optix::Material material_;
		optix::Geometry geometry_;
		mat4 transform_;

		void ProcessNode(aiNode* node, const aiScene* scene);
		MeshData TranslateMesh(aiMesh* mesh, const aiScene* scene) const;

		[[nodiscard]] optix::Geometry CreatePlaneGeometry(const int tess_u, const int tess_v, const int up_axis) const;
		[[nodiscard]] optix::Geometry CreateSphereGeometry(const int tess_u, const int tess_v, const float radius,
											 const float max_theta) const;
		static optix::Geometry CreateGeometry(const std::vector<VertexAttributes>& attributes,
											  const std::vector<unsigned>& indices);
	};
}
