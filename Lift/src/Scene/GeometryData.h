#pragma once
#include "renderer/BufferView.h"
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

namespace lift {
	struct GeometryData {
		enum Type {
			TRIANGLE_MESH = 0,
			SPHERE
		};

		struct TriangleMesh {
			BufferView<glm::ivec3> indices;
			BufferView<glm::vec3> positions;
			BufferView<glm::vec3> normals;
			BufferView<glm::vec2> tex_coords;
		};

		struct Sphere {
			glm::vec3 center;
			float radius;
		};

		Type type;

		union {
			TriangleMesh triangle_mesh;
			Sphere sphere;
		};
	};
}
