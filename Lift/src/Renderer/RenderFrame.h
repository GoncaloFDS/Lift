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

		[[nodiscard]] optix::Buffer& GetBufferOutput() { return output_buffer; }
		unsigned GetTextureId() const { return texture_->id; }

	private:
		std::unique_ptr<PixelBuffer> output_pixel_buffer_;
		std::unique_ptr<Texture> texture_;
		optix::Buffer output_buffer;

		uint32_t width_;
		uint32_t height_;

	};
}
