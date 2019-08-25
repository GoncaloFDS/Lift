#pragma once

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "renderer/Renderer.h"

namespace lift {
	struct Mesh {
	public:
		Mesh() : transform_(1.0f) { }

		Mesh(const std::string& path);

		[[nodiscard]] const mat4& Transform() const { return transform_; }
		void SetTransform(const mat4& transform) { transform_ = transform; }

		std::vector<vec3> vertices;
		std::vector<vec3> normals;
		std::vector<vec2> tex_coords;
		std::vector<ivec3> indices;
		vec3 diffuse{};
		
	protected:
		mat4 transform_;

		void ProcessNode(aiNode* node, const aiScene* scene);

	};
}
