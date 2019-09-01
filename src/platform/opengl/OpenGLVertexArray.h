#pragma once

#include "Renderer/VertexArray.h"

namespace lift {

	class OpenGLVertexArray : public VertexArray {
	public:

		OpenGLVertexArray();
		virtual ~OpenGLVertexArray() = default;

		void Bind() const override;
		void Unbind() const override;

		void AddVertexBuffer(const std::shared_ptr<VertexBuffer>& vertex_buffer) override;
		void SetIndexBuffer(const std::shared_ptr<IndexBuffer>& index_buffer) override;

		const std::vector<std::shared_ptr<VertexBuffer>>& GetVertexBuffers() const override { return vertex_buffers_; }
		const std::shared_ptr<IndexBuffer>& GetIndexBuffer() const override { return index_buffer_; }

	private:
		uint32_t renderer_id_ {};
		std::vector<std::shared_ptr<VertexBuffer>> vertex_buffers_ {};
		std::shared_ptr<IndexBuffer> index_buffer_;
	};
}
