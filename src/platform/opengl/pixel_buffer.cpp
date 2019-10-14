#include "pch.h"
#include "pixel_buffer.h"
#include "glad/glad.h"

lift::PixelBuffer::PixelBuffer(const float size) {
    GL_CHECK(glGenBuffers(1, &id));
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id));
    GL_CHECK(glBufferData(GL_PIXEL_UNPACK_BUFFER, GLsizeiptr(size), nullptr, GL_STREAM_READ));
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
}

void lift::PixelBuffer::bind() const {
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id));
}

void lift::PixelBuffer::unbind() const {
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
}

void lift::PixelBuffer::resize(const unsigned size) {
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id));
    GL_CHECK(glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
}
