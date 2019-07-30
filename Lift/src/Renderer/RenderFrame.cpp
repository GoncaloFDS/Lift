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

	float quad_vertices [4 * 5] = {
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
	};

	vertex_array_.reset(VertexArray::Create());
	std::shared_ptr<VertexBuffer> vertex_buffer{};
	vertex_buffer.reset(VertexBuffer::Create(quad_vertices, sizeof(quad_vertices)));

	vertex_buffer->SetLayout({
		{ShaderDataType::Float3, "a_Position"},
		{ShaderDataType::Float2, "a_Uv"}
	});

	vertex_array_->AddVertexBuffer(vertex_buffer);
	uint32_t indices[6] = {0, 1, 2, 0, 2, 3};
	std::shared_ptr<IndexBuffer> index_buffer;
	index_buffer.reset(IndexBuffer::Create(indices, sizeof(indices) / sizeof(uint32_t)));
	vertex_array_->SetIndexBuffer(index_buffer);

	shader_ = std::make_unique<Shader>("res/shaders/texture_quad");
	shader_->Bind();
	shader_->SetUniform1i("u_Texture", 0);

}

void lift::RenderFrame::Bind() const {
	texture_->Bind();
	output_pixel_buffer_->Bind();
	shader_->Bind();
	shader_->SetTexImage2D(width_, height_);
}

void lift::RenderFrame::Resize(const uint32_t width, const uint32_t height) {
	width_ = width;
	height_ = height;
	output_buffer->setSize(width, height);
	output_pixel_buffer_->Resize(unsigned(output_buffer->getElementSize()) * width * height);
}
