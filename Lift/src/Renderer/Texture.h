#pragma once

namespace lift {

	enum class TextureType {
		None = 0,
		Diffuse
	};

	struct Texture {
		unsigned int id{};
		ivec2 resolution{};

		Texture(const std::string& path);
		Texture();
		void Bind(unsigned int slot = 0) const;
		void Unbind();
		void SetData();
		auto Data() { return data_.data(); }

		void Resize(const ivec2& size) {
			resolution = size;
			data_.resize(size.x * size.y);
		}

	private:
		std::vector<uint32_t> data_;
	};
}
