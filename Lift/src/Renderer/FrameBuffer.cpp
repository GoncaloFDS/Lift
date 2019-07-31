#include "pch.h"
#include "FrameBuffer.h"
#include "Texture.h"
#include "glad/glad.h"


lift::FrameBuffer::FrameBuffer() : renderer_id_(0) {
	glGenFramebuffers(1, &renderer_id_);
}

lift::FrameBuffer::~FrameBuffer() {
	glDeleteFramebuffers(1, &renderer_id_);
}

void lift::FrameBuffer::Bind() const {
	glBindFramebuffer(GL_FRAMEBUFFER, renderer_id_);
}

void lift::FrameBuffer::Unbind() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void lift::FrameBuffer::BindTexture(const Texture texture) {
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture.id, 0);
}
