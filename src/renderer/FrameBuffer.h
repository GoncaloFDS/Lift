#pragma once
namespace lift {
	struct Texture;

	class FrameBuffer {
	public:
		FrameBuffer();
		~FrameBuffer();

		void Bind() const;
		void Unbind();
		void BindTexture(Texture texture);

	private:
		unsigned renderer_id_;
	};
}
