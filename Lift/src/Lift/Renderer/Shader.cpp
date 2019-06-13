#include "pch.h"
#include "Shader.h"

#include <glad/glad.h>

namespace lift {

	Shader::Shader(const std::string& vertex_src, const std::string& fragment_src) {
		// Create and empty vertex shader handle
		const GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);

		// Send the vertex shader source
		const GLchar* source = vertex_src.c_str();
		glShaderSource(vertex_shader, 1, &source, nullptr);

		glCompileShader(vertex_shader);

		GLint is_compiled = 0;
		glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &is_compiled);
		if (is_compiled == GL_FALSE) {
			GLint max_length = 0;
			glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &max_length);

			std::vector<GLchar> info_log(max_length);
			glGetShaderInfoLog(vertex_shader, max_length, &max_length, &info_log[0]);

			glDeleteShader(vertex_shader);

			LF_CORE_ERROR("{0}", info_log.data());
			LF_CORE_ASSERT(false, "Vertex shader compilation failed")
			return;
		}

		const GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
		source = fragment_src.c_str();
		glShaderSource(fragment_shader, 1, &source, nullptr);

		glCompileShader(fragment_shader);

		glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &is_compiled);
		if (is_compiled == GL_FALSE)
		{
			GLint maxLength = 0;
			glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &maxLength);

			// The maxLength includes the NULL character
			std::vector<GLchar> infoLog(maxLength);
			glGetShaderInfoLog(fragment_shader, maxLength, &maxLength, &infoLog[0]);

			// We don't need the shader anymore.
			glDeleteShader(fragment_shader);
			// Either of them. Don't leak shaders.
			glDeleteShader(vertex_shader);

			LF_CORE_ERROR("{0}", infoLog.data());
			LF_CORE_ASSERT(false, "Fragment shader compilation failure!");
			return;
		}

		// Vertex and fragment shaders are successfully compiled.
		// Now time to link them together into a program.
		// Get a program object.
		renderer_id_ = glCreateProgram();
		GLuint program = renderer_id_;

		// Attach our shaders to our program
		glAttachShader(program, vertex_shader);
		glAttachShader(program, fragment_shader);

		// Link our program
		glLinkProgram(program);

		// Note the different functions here: glGetProgram* instead of glGetShader*.
		GLint is_linked = 0;
		glGetProgramiv(program, GL_LINK_STATUS, static_cast<int*>(&is_linked));
		if (is_linked == GL_FALSE)
		{
			GLint maxLength = 0;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

			// The maxLength includes the NULL character
			std::vector<GLchar> info_log(maxLength);
			glGetProgramInfoLog(program, maxLength, &maxLength, &info_log[0]);

			// We don't need the program anymore.
			glDeleteProgram(program);
			// Don't leak shaders either.
			glDeleteShader(vertex_shader);
			glDeleteShader(fragment_shader);

			LF_CORE_ERROR("{0}", info_log.data());
			LF_CORE_ASSERT(false, "Shader link failure!");
			return;
		}

		// Always detach shaders after a successful link.
		glDetachShader(program, vertex_shader);
		glDetachShader(program, fragment_shader);

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
}
