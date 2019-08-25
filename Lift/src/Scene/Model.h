#pragma once
#include "Mesh.h"
#include "core/Profiler.h"

namespace lift {
	struct Model {
		Model(const std::string& path);

		void ProcessNode(aiNode* node, const aiScene* scene);

	public:
		std::vector<Mesh> meshes;
	};
}
