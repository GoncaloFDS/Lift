#pragma once

namespace lift {

	enum class TextureType {
		kNone = 0, kDiffuse
	};

	struct Texture {
		unsigned int id;
		vec2 resolution;

		Texture(const std::string& path);

		void Bind(unsigned int slot = 0) const;
		void Unbind();

	};
}
