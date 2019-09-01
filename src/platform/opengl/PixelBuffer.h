#pragma once

namespace lift {

	struct  PixelBuffer {
		PixelBuffer(float size);
		virtual ~PixelBuffer() = default;

		virtual void Bind() const;
		virtual void Unbind() const;

		virtual void Resize(const unsigned size);

		unsigned id{};
	};
}
