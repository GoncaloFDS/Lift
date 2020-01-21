#include "texture.h"
#include "core/stb_image_impl.h"
#include <chrono>
#include <iostream>
#include <core.h>
#include <pch.h>

namespace assets {

Texture Texture::LoadTexture(const std::string& filename, const vulkan::SamplerConfig& sampler_config) {
    LF_WARN("Loading texture {0}", filename);
    const auto timer = std::chrono::high_resolution_clock::now();

    // Load the texture in normal host memory.
    int width, height, channels;
    const auto pixels = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    LF_ASSERT(pixels, "failed to load texture image '{0}'", filename);

    const auto elapsed = std::chrono::duration<float, std::chrono::seconds::period>(
        std::chrono::high_resolution_clock::now() - timer).count();
    std::cout << "(" << width << " x " << height << " x " << channels << ") ";
    std::cout << elapsed << "s" << std::endl;

    return Texture(width, height, channels, pixels);
}

Texture::Texture(int width, int height, int channels, unsigned char* const pixels) :
    width_(width),
    height_(height),
    channels_(channels),
    pixels_(pixels, stbi_image_free) {
}

}
