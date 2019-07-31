#include "pch.h"
#include "RenderFrame.h"
#include "Shader.h"
#include "Platform/Optix/OptixContext.h"


void lift::RenderFrame::Init(const unsigned width, const unsigned height) {
	width_ = width;
	height_ = height;
	output_pixel_buffer_ = std::make_unique<PixelBuffer>(
		static_cast<float>(width_ * height_) * sizeof(float) * 4);
	texture_ = std::make_unique<Texture>();
	output_buffer = OptixContext::Get()->createBufferFromGLBO(RT_BUFFER_OUTPUT, output_pixel_buffer_->id);
	output_buffer->setFormat(RT_FORMAT_FLOAT4); //RGBA32F
	output_buffer->setSize(width_, height_);

}

void lift::RenderFrame::Bind() const {
	texture_->Bind();
	output_pixel_buffer_->Bind();
	Shader::SetTexImage2D(width_, height_);
}

void lift::RenderFrame::Resize(const uint32_t width, const uint32_t height) {
	width_ = width;
	height_ = height;
	output_buffer->setSize(width, height);
	output_pixel_buffer_->Resize(unsigned(output_buffer->getElementSize()) * width * height);
}
