#include "texture.h"
#include "core/stb_image_impl.h"
#include <chrono>
#include <core.h>

namespace assets {

Texture Texture::loadTexture(const std::string &filename) {
    LF_WARN("Loading texture {0}", filename);
    const auto timer = std::chrono::high_resolution_clock::now();

    // Load the texture in normal host memory.
    int width, height, channels;
    const auto pixels = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    LF_ASSERT(pixels, "failed to load texture image '{0}'", filename);

    const auto elapsed =
        std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::high_resolution_clock::now() - timer)
            .count();
    LF_INFO("Loaded Texture {0} in {1}, -> {2}x{3} with {4} channels", filename, elapsed, width, height, channels);

    return Texture(width, height, channels, pixels);
}

Texture::Texture(int width, int height, int channels, unsigned char *const pixels) :
    width_(width), height_(height), channels_(channels), pixels_(pixels, stbi_image_free) {
}

}  // namespace assets
