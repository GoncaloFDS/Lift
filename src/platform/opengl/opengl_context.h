#pragma once

#include <glad/glad.h>
#include <core/os/window.h>
#include "renderer/graphics_context.h"

struct GLFWwindow;

namespace lift {
class OpenGLContext : public GraphicsContext {
public:

	explicit OpenGLContext(std::shared_ptr<Window> window_handle,
						   BufferImageFormat format = BufferImageFormat::UNSIGNED_BYTE_4);
	void init() override;
	void display(ivec2 screen_res, ivec2 frame_buffer_res, uint32_t pbo);
	void swapBuffers() override;

private:
	GLFWwindow* window_handle_;
	GLuint render_tex_ = 0u;
	GLuint program_ = 0u;
	GLint render_tex_uniform_loc_ = -1;
	GLuint quad_vertex_buffer_ = 0;

	BufferImageFormat image_format_;

	static const std::string s_vert_source;
	static const std::string s_frag_source;
};

auto createGLProgram(const std::string& vert_source, const std::string& frag_source) -> GLuint;
auto getGLUniformLocation(GLuint program, const std::string& name) -> GLint;
auto pixelFormatSize(lift::BufferImageFormat format) -> size_t;

}
