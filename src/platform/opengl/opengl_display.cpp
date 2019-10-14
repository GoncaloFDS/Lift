#include "pch.h"
#include "opengl_display.h"

namespace {

GLuint createGLShader(const std::string& source, GLuint shader_type) {
    GLuint shader = glCreateShader(shader_type);
    {
        const GLchar* source_data = reinterpret_cast<const GLchar*>( source.data());
        glShaderSource(shader, 1, &source_data, nullptr);
        glCompileShader(shader);

        GLint is_compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
        if (is_compiled == GL_FALSE) {
            GLint max_length = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length);

            std::string info_log(max_length, '\0');
            GLchar* info_log_data = reinterpret_cast<GLchar*>( &info_log[0]);
            glGetShaderInfoLog(shader, max_length, nullptr, info_log_data);

            glDeleteShader(shader);
            std::cerr << "Compilation of shader failed: " << info_log << std::endl;

            return 0;
        }
    }

    return shader;
}

GLuint createGLProgram(
    const std::string& vert_source,
    const std::string& frag_source
) {
    GLuint vert_shader = createGLShader(vert_source, GL_VERTEX_SHADER);
    if (vert_shader == 0)
        return 0;

    GLuint frag_shader = createGLShader(frag_source, GL_FRAGMENT_SHADER);
    if (frag_shader == 0) {
        glDeleteShader(vert_shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);

    GLint is_linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
    if (is_linked == GL_FALSE) {
        GLint max_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &max_length);

        std::string info_log(max_length, '\0');
        GLchar* info_log_data = reinterpret_cast<GLchar*>( &info_log[0]);
        glGetProgramInfoLog(program, max_length, nullptr, info_log_data);
        std::cerr << "Linking of program failed: " << info_log << std::endl;

        glDeleteProgram(program);
        glDeleteShader(vert_shader);
        glDeleteShader(frag_shader);

        return 0;
    }

    glDetachShader(program, vert_shader);
    glDetachShader(program, frag_shader);

    return program;
}

GLint getGLUniformLocation(GLuint program, const std::string& name) {
    GLint loc = glGetUniformLocation(program, name.c_str());
    return loc;
}

}
//-----------------------------------------------------------------------------
//
// GLDisplay implementation
//
//-----------------------------------------------------------------------------

const std::string lift::OpenGLDisplay::s_vert_source = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

const std::string lift::OpenGLDisplay::s_frag_source = R"(
#version 330 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";

lift::OpenGLDisplay::OpenGLDisplay(lift::BufferImageFormat image_format) : image_format_(image_format) {
    GLuint m_vertex_array;
    GL_CHECK(glGenVertexArrays(1, &m_vertex_array));
    GL_CHECK(glBindVertexArray(m_vertex_array));

    program_ = createGLProgram(s_vert_source, s_frag_source);
    render_tex_uniform_loc_ = getGLUniformLocation(program_, "render_tex");

    GL_CHECK(glGenTextures(1, &render_tex_));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, render_tex_));

    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
    };

    GL_CHECK(glGenBuffers(1, &quad_vertex_buffer_));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, quad_vertex_buffer_));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER,
                          sizeof(g_quad_vertex_buffer_data),
                          g_quad_vertex_buffer_data,
                          GL_STATIC_DRAW
    )
    );

}

void lift::OpenGLDisplay::display(ivec2 screen_res, ivec2 frame_buffer_res, uint32_t pbo) {
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL_CHECK(glViewport(0, 0, frame_buffer_res.x, frame_buffer_res.y));

    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    GL_CHECK(glUseProgram(program_));

    // Bind our texture in Texture Unit 0
    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, render_tex_));
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo));

    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 4)); // TODO!!!!!!

    size_t elmt_size = pixelFormatSize(image_format_);
    if (elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if (elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if (elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (image_format_ == BufferImageFormat::UNSIGNED_BYTE_4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_res.x, screen_res.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    else if (image_format_ == BufferImageFormat::FLOAT_3)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, screen_res.x, screen_res.y, 0, GL_RGB, GL_FLOAT, nullptr);

    else if (image_format_ == BufferImageFormat::FLOAT_4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screen_res.x, screen_res.y, 0, GL_RGBA, GL_FLOAT, nullptr);

    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
    GL_CHECK(glUniform1i(render_tex_uniform_loc_, 0));

    // 1st attribute buffer : vertices
    GL_CHECK(glEnableVertexAttribArray(0));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, quad_vertex_buffer_));
    GL_CHECK(glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*) 0            // array buffer offset
    )
    );

    // Draw the triangles !
    GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 6)); // 2*3 indices starting at 0 -> 2 triangles

    GL_CHECK(glDisableVertexAttribArray(0));
}
