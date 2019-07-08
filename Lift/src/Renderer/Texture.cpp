#include "pch.h"
#include "Texture.h"
#include "glad/glad.h"

#include "stb_image.h"

namespace lift {

	Texture::Texture(const std::string& path) {
		stbi_set_flip_vertically_on_load(1);

		OPENGL_CALL(glGenTextures(1, &id));

		int width, height, component_count;
		unsigned char* data = stbi_load(path.c_str(), &width, &height, &component_count, 0);
		if (data) {
			GLenum format = 0;
			if (component_count == 1)
				format = GL_RED;
			if (component_count == 3)
				format = GL_RGB;
			if (component_count == 4)
				format = GL_RGBA;

			OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, id));
			OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data));
			OPENGL_CALL(glGenerateMipmap(GL_TEXTURE_2D));

			OPENGL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
			OPENGL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
			OPENGL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
			OPENGL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

		}
		else {
			LF_CORE_ERROR("Texture {0} failed to load", path);
		}
		stbi_image_free(data);
		resolution = vec2(static_cast<float>(width), static_cast<float>(height));
	}

	Texture::Texture(optix::Buffer buffer) {
		OPENGL_CALL(glGenTextures(1, &id));
		RTsize buffer_width, buffer_height;
		buffer->getSize(buffer_width, buffer_height);
		const uint32_t width = static_cast<int>(buffer_width);
		const uint32_t height = static_cast<int>(buffer_height);
		RTformat buffer_format = buffer->getFormat();

		GLboolean use_rgb = GL_FALSE;
		OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, id));

		OPENGL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		OPENGL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));

		OPENGL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
		OPENGL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

		OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, id));

		const unsigned pbo_id = buffer->getGLBOId();
		GLvoid* image_data = nullptr;
		if (pbo_id) {
			OPENGL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id));
		}
		else {
			image_data = buffer->map(0, RT_BUFFER_MAP_READ);
		}

		const RTsize element_size = buffer->getElementSize();
		if (element_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
		else if (element_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		else if (element_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
		else glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data));

		if (pbo_id){
			OPENGL_CALL(GL_PIXEL_UNPACK_BUFFER, 0);
		}
		else
			buffer->unmap();
	}

	void Texture::Bind(unsigned int slot) const {
		OPENGL_CALL(glActiveTexture(GL_TEXTURE0 + slot));
		OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, id));
	}

	void Texture::Unbind() {
		OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, 0));
	}


}
