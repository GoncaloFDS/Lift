#pragma once

#include "Renderer/Buffer.h"

namespace lift {

	class OpenGLVertexBuffer : public VertexBuffer {
	public:
		OpenGLVertexBuffer(float* vertices, uint32_t size);
		virtual ~OpenGLVertexBuffer();

		void Bind() const override;
		void Unbind() const override;

		const BufferLayout& GetLayout() const override { return layout_; }
		void SetLayout(const BufferLayout& layout) override { layout_ = layout; }

	private:
		uint32_t renderer_id_;
		BufferLayout layout_;
	};

	class OpenGLIndexBuffer : public IndexBuffer {
	public:
		OpenGLIndexBuffer(uint32_t* indices, uint32_t count);
		virtual ~OpenGLIndexBuffer();

		void Bind() const override;
		void Unbind() const override;

		uint32_t GetCount() const override;
	private:
		uint32_t renderer_id_;
		uint32_t count_;
	};

}
