#pragma once
#include "VertexArray.h"
#include "Shader.h"
#include "Platform/OpenGL/PixelBuffer.h"
#include "Texture.h"


namespace lift {

	class RenderFrame {
	public:
		RenderFrame() = default;

		void Init(unsigned width, unsigned height);
		void Bind() const;

		void Resize(uint32_t width, uint32_t height);

		[[nodiscard]] std::shared_ptr<VertexArray> GetVertexArray() const { return vertex_array_; }

		[[nodiscard]] optix::Buffer& GetBufferOutput() { return output_buffer; }

		unsigned GetTextureId() const { return texture_->id; }

	private:
		std::shared_ptr<VertexArray> vertex_array_;
		std::shared_ptr<Shader> shader_;
		std::unique_ptr<PixelBuffer> output_pixel_buffer_;
		std::unique_ptr<Texture> texture_;
		optix::Buffer output_buffer;

		uint32_t width_;
		uint32_t height_;

	};
}
