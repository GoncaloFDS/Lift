#pragma once

#include "vulkan/sampler.h"
#include <memory>
#include <string>

namespace assets {
class Texture final {
public:

    static Texture loadTexture(const std::string& filename, const vulkan::SamplerConfig& sampler_config);

    Texture& operator=(const Texture&) = delete;
    Texture& operator=(Texture&&) = delete;

    Texture(Texture&&) = default;
    ~Texture() = default;

    [[nodiscard]] const unsigned char* pixels() const { return pixels_.get(); }
    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }

private:

    Texture(int width, int height, int channels, unsigned char* pixels);

    vulkan::SamplerConfig sampler_config_;
    int width_;
    int height_;
    int channels_;
    std::unique_ptr<unsigned char, void (*)(void*)> pixels_;
};

}
