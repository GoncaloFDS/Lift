#pragma once
#include <optix_world.h>

namespace lift {

	enum class TextureType {
		kNone = 0, kDiffuse
	};

	struct Texture {
		unsigned int id;
		vec2 resolution;

		Texture(const std::string& path);
		Texture(optix::Buffer buffer);
		void Bind(unsigned int slot = 0) const;
		void Unbind();

	};
}
