#include "pch.h"
#include "texture.h"
#include "glad/glad.h"

//#include "stb_image.h"

lift::Texture::Texture(const std::string& path) {
//	stbi_set_flip_vertically_on_load(1);
//
//	GL_CHECK(glGenTextures(1, &id));
//
//	int width, height, component_count;
//	unsigned char* data = stbi_load(path.c_str(), &width, &height, &component_count, 0);
//	if (data) {
//		GLenum format = 0;
//		if (component_count == 1)
//			format = GL_RED;
//		if (component_count == 3)
//			format = GL_RGB;
//		if (component_count == 4)
//			format = GL_RGBA;
//
//		GL_CHECK(glBindTexture(GL_TEXTURE_2D, id));
//		GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data));
//		GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
//
//		GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
//		GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
//		GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
//		GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
//
//	}
//	else {
//		LF_ERROR("Texture {0} failed to load", path);
//	}
//	stbi_image_free(data);
//	resolution = vec2(static_cast<float>(width), static_cast<float>(height));
}

lift::Texture::Texture() : resolution(0, 0) {
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

}

void lift::Texture::bind(unsigned int slot) const {
    GL_CHECK(glActiveTexture(GL_TEXTURE0 + slot));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, id));
}

void lift::Texture::unbind() {
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
}

void lift::Texture::setData() {
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, id));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution.x, resolution.y,
                          0, GL_RGBA, GL_UNSIGNED_BYTE, data_.data()));
}