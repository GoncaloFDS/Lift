#include "pch.h"
#include "Shader.h"

#include <glad/glad.h>
#include <fstream>
#include <sstream>

lift::Shader::Shader(const std::string& file_path)
	: renderer_id_{0}, file_path_{file_path} {

	const ShaderProgramSource shader_program = ParseShader(file_path);
	renderer_id_ = CreateShader(shader_program.vertex_source, shader_program.fragment_source);
}

lift::Shader::~Shader() {
	glDeleteProgram(renderer_id_);
}


void lift::Shader::Bind() const {
	glUseProgram(renderer_id_);
}

void lift::Shader::Unbind() const {
	glUseProgram(0);
}

void lift::Shader::SetUniform1i(const std::string& name, int value) {
	OPENGL_CALL(glUniform1i(GetUniformLocation(name), value));
}

void lift::Shader::SetUniform1f(const std::string& name, float value) {
	OPENGL_CALL(glUniform1f(GetUniformLocation(name), value));
}

void lift::Shader::SetTexImage2D(uint32_t width, uint32_t height) {
	OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr));
}

ShaderProgramSource lift::Shader::ParseShader(const std::string& file_path) const {
	std::string line;
	std::stringstream string_stream[2];

	std::ifstream vertex_stream(file_path + ".vert");
	while (std::getline(vertex_stream, line))
		string_stream[0] << line << '\n';

	std::ifstream fragment_stream(file_path + ".frag");
	while (std::getline(fragment_stream, line))
		string_stream[1] << line << '\n';

	return {string_stream[0].str(), string_stream[1].str()};

}

unsigned lift::Shader::CreateShader(const std::string& vertex_source, const std::string& fragment_source) {
	const unsigned int program = glCreateProgram();
	const unsigned int vertex_shader = CompileShader(GL_VERTEX_SHADER, vertex_source);
	const unsigned int fragment_shader = CompileShader(GL_FRAGMENT_SHADER, fragment_source);

	OPENGL_CALL(glAttachShader(program, vertex_shader));
	OPENGL_CALL(glAttachShader(program, fragment_shader));
	OPENGL_CALL(glLinkProgram(program));
	OPENGL_CALL(glValidateProgram(program));

	OPENGL_CALL(glDeleteShader(vertex_shader));
	OPENGL_CALL(glDeleteShader(fragment_shader));

	return program;
}

unsigned lift::Shader::CompileShader(const unsigned int type, const std::string& source) {
	const unsigned int shader_id = glCreateShader(type);
	const GLchar* src = source.c_str();

	OPENGL_CALL(glShaderSource(shader_id, 1, &src, 0));
	OPENGL_CALL(glCompileShader(shader_id));

	int is_compiled = 0;
	OPENGL_CALL(glGetShaderiv(shader_id, GL_COMPILE_STATUS, &is_compiled));
	if (is_compiled == GL_FALSE) {
		int buf_size = 0;
		OPENGL_CALL(glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &buf_size));

		std::vector<GLchar> info_log(buf_size);
		OPENGL_CALL(glGetShaderInfoLog(shader_id, buf_size, &buf_size, &info_log[0]));

		LF_CORE_ERROR("Failed to Compile {0}", (type == GL_VERTEX_SHADER ? "Vertex Shader" : "Fragment Shader"));
		LF_CORE_ERROR("{0}", info_log.data());
		OPENGL_CALL(glDeleteShader(shader_id));
		return 0;
	}

	return shader_id;
}

int lift::Shader::GetUniformLocation(const std::string& name) {
	if (uniform_location_cache_.find(name) != uniform_location_cache_.end())
		return uniform_location_cache_[name];

	const int location = glGetUniformLocation(renderer_id_, name.c_str());
	OPENGL_CALL(name);
	if (location == -1)
		LF_CORE_WARN("Uniform {0} is not defined on shader {1}", name, file_path_);
	return location;
}
