#include "texture.h"
#include "core/stb_image_impl.h"
#include <chrono>
#include <core.h>
#include <core/profiler.h>

namespace assets {

Texture Texture::loadTexture(const std::string& filename) {
    LF_INFO("Loading texture {0}", filename);
    Profiler profiler("Loading texture took");

    // Load the texture in normal host memory.
    int width, height, channels;
    const auto pixels = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    LF_ASSERT(pixels, "failed to load texture image '{0}'", filename);

    return Texture(width, height, channels, pixels);
}

Texture::Texture(int width, int height, int channels, unsigned char* const pixels) :
    width_(width), height_(height), channels_(channels), pixels_(pixels, stbi_image_free) {
}

}  // namespace assets
