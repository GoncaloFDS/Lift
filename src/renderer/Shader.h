#pragma once

#include <string>

struct ShaderProgramSource {
	std::string vertex_source;
	std::string fragment_source;
};

namespace lift {
	class Shader {
	public:
		Shader(const std::string& file_path);
		~Shader();

		void Bind() const;
		void Unbind() const;

		void SetUniform1i(const std::string& name, int value);
		void SetUniform1f(const std::string& name, float value);

		static void SetTexImage2D(uint32_t width, uint32_t height);

	private:
		ShaderProgramSource ParseShader(const std::string& file_path) const;
		static unsigned int CreateShader(const std::string& vertex_source, const std::string& fragment_source);
		static unsigned CompileShader(unsigned int type, const std::string& source);
		int GetUniformLocation(const std::string& name);


		uint32_t renderer_id_;
		std::string file_path_;
		std::unordered_map<std::string, int> uniform_location_cache_;
	};
}
