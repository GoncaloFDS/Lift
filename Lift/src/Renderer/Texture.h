#pragma once

namespace lift {

	enum class TextureType {
		None = 0,
		Diffuse
	};

	struct Texture {
		unsigned int id{};
		vec2 resolution{};

		Texture(const std::string& path);
		Texture();
		void Bind(unsigned int slot = 0) const;
		void Unbind();

	};
}
