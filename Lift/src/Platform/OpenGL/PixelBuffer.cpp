#include "pch.h"
#include "PixelBuffer.h"
#include "glad/glad.h"

lift::PixelBuffer::PixelBuffer(const float size) {
	OPENGL_CALL(glGenBuffers(1, &id));
	OPENGL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id));
	OPENGL_CALL(glBufferData(GL_PIXEL_UNPACK_BUFFER, GLsizeiptr(size), nullptr, GL_STREAM_READ));
	OPENGL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
}

void lift::PixelBuffer::Bind() const {
	OPENGL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id));
}

void lift::PixelBuffer::Unbind() const {
	OPENGL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
}

void lift::PixelBuffer::Resize(const unsigned size) {
	OPENGL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id));
	OPENGL_CALL(glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_STREAM_DRAW));
	OPENGL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
}
