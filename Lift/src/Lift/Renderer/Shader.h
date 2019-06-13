#pragma once

#include <string>

namespace lift {
	class Shader {
	public:
		Shader(const std::string& vertex_src, const std::string& fragment_src);
		~Shader();

		void Bind() const;
		void Unbind() const;

	private:
		uint32_t renderer_id_;
	};
}
