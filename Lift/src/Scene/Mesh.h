#pragma once

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "renderer/Renderer.h"

namespace lift {
	class Mesh : public TriangleMesh {
	public:
		Mesh() : transform_(1.0f) {
		}

		Mesh(const std::string& path);

		[[nodiscard]] const mat4& Transform() const { return transform_; }
		void SetTransform(const mat4& transform) { transform_ = transform; }

	protected:
		mat4 transform_;

		void ProcessNode(aiNode* node, const aiScene* scene);

	};
}
