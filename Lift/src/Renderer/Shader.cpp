#include "pch.h"
#include "Shader.h"

#include <glad/glad.h>
#include <fstream>
#include <sstream>

namespace lift {

//	Shader::Shader(const std::string& vertex_src, const std::string& fragment_src) {
//		// Create and empty vertex shader handle
//		const GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
//
//		// Send the vertex shader source
//		const GLchar* source = vertex_src.c_str();
//		glShaderSource(vertex_shader, 1, &source, nullptr);
//
//		glCompileShader(vertex_shader);
//
//		GLint is_compiled = 0;
//		glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &is_compiled);
//		if (is_compiled == GL_FALSE) {
//			GLint max_length = 0;
//			glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &max_length);
//
//			std::vector<GLchar> info_log(max_length);
//			glGetShaderInfoLog(vertex_shader, max_length, &max_length, &info_log[0]);
//
//			glDeleteShader(vertex_shader);
//
//			LF_CORE_ERROR("{0}", info_log.data());
//			LF_CORE_ASSERT(false, "Vertex shader compilation failed")
//			return;
//		}
//
//		const GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
//		source = fragment_src.c_str();
//		glShaderSource(fragment_shader, 1, &source, nullptr);
//
//		glCompileShader(fragment_shader);
//
//		glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &is_compiled);
//		if (is_compiled == GL_FALSE) {
//			GLint maxLength = 0;
//			glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &maxLength);
//
//			// The maxLength includes the NULL character
//			std::vector<GLchar> infoLog(maxLength);
//			glGetShaderInfoLog(fragment_shader, maxLength, &maxLength, &infoLog[0]);
//
//			// We don't need the shader anymore.
//			glDeleteShader(fragment_shader);
//			// Either of them. Don't leak shaders.
//			glDeleteShader(vertex_shader);
//
//			LF_CORE_ERROR("{0}", infoLog.data());
//			LF_CORE_ASSERT(false, "Fragment shader compilation failure!");
//			return;
//		}
//
//		// Vertex and fragment shaders are successfully compiled.
//		// Now time to link them together into a program.
//		// Get a program object.
//		renderer_id_ = glCreateProgram();
//		GLuint program = renderer_id_;
//
//		// Attach our shaders to our program
//		glAttachShader(program, vertex_shader);
//		glAttachShader(program, fragment_shader);
//
//		// Link our program
//		glLinkProgram(program);
//
//		// Note the different functions here: glGetProgram* instead of glGetShader*.
//		GLint is_linked = 0;
//		glGetProgramiv(program, GL_LINK_STATUS, static_cast<int*>(&is_linked));
//		if (is_linked == GL_FALSE) {
//			GLint maxLength = 0;
//			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
//
//			// The maxLength includes the NULL character
//			std::vector<GLchar> info_log(maxLength);
//			glGetProgramInfoLog(program, maxLength, &maxLength, &info_log[0]);
//
//			// We don't need the program anymore.
//			glDeleteProgram(program);
//			// Don't leak shaders either.
//			glDeleteShader(vertex_shader);
//			glDeleteShader(fragment_shader);
//
//			LF_CORE_ERROR("{0}", info_log.data());
//			LF_CORE_ASSERT(false, "Shader link failure!");
//			return;
//		}
//
//		// Always detach shaders after a successful link.
//		glDetachShader(program, vertex_shader);
//		glDetachShader(program, fragment_shader);
//
//	}

	Shader::Shader(const std::string& file_path)
		: renderer_id_{0} {

		ShaderProgramSource shader_program = ParseShader(file_path);
		renderer_id_ = CreateShader(shader_program.vertex_source, shader_program.fragment_source);
	}

	Shader::~Shader() {
		glDeleteProgram(renderer_id_);
	}


	void Shader::Bind() const {
		glUseProgram(renderer_id_);
	}

	void Shader::Unbind() const {
		glUseProgram(0);
	}

	ShaderProgramSource Shader::ParseShader(const std::string& file_path) const {

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

	unsigned Shader::CreateShader(const std::string& vertex_source, const std::string& fragment_source) {
		unsigned int program = glCreateProgram();
		unsigned int vertex_shader = CompileShader(GL_VERTEX_SHADER, vertex_source);
		unsigned int fragment_shader = CompileShader(GL_FRAGMENT_SHADER, fragment_source);

		OPENGL_CALL(glAttachShader(program, vertex_shader));
		OPENGL_CALL(glAttachShader(program, fragment_shader));
		OPENGL_CALL(glLinkProgram(program));
		OPENGL_CALL(glValidateProgram(program));

		OPENGL_CALL(glDeleteShader(vertex_shader));
		OPENGL_CALL(glDeleteShader(fragment_shader));

		return program;
	}

	unsigned Shader::CompileShader(const unsigned int type, const std::string& source) {
		unsigned int shader_id = glCreateShader(type);
		const GLchar* src = source.c_str();

		OPENGL_CALL(glShaderSource(shader_id, 1, &src, 0));
		OPENGL_CALL(glCompileShader(shader_id));

		int is_compiled = 0;
		OPENGL_CALL(glGetShaderiv(shader_id, GL_COMPILE_STATUS, &is_compiled));
		if (is_compiled == GL_FALSE) {
			int lenght = 0;
			OPENGL_CALL(glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &lenght));
			
			std::vector<GLchar> info_log(lenght);
			OPENGL_CALL(glGetShaderInfoLog(shader_id, lenght, &lenght, &info_log[0]));

			LF_CORE_ERROR("Failed to Compile {0}", (type == GL_VERTEX_SHADER ? "Vertex Shader" : "Fragment Shader"));
			LF_CORE_ERROR("{0}", info_log.data());
			OPENGL_CALL(glDeleteShader(shader_id));
			return 0;
		}

		return shader_id;
	}

	int Shader::GetUniformLocation(const std::string& name) {
		return 0;
	}
}
