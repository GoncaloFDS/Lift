//
// Created by gonca on 07/10/2019.
//

#pragma once
#include <glad/glad.h>
#include "opengl_renderer_api.h"

namespace lift {

class OpenGLDisplay {
 public:
    explicit OpenGLDisplay(BufferImageFormat format = BufferImageFormat::UNSIGNED_BYTE_4);

    void display(ivec2 screen_res, ivec2 frame_buffer_res, uint32_t pbo);

 private:
    GLuint render_tex_ = 0u;
    GLuint program_ = 0u;
    GLint render_tex_uniform_loc_ = -1;
    GLuint quad_vertex_buffer_ = 0;

    BufferImageFormat image_format_;

    static const std::string s_vert_source;
    static const std::string s_frag_source;
};

}
