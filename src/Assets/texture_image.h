#pragma once

#include <memory>

namespace vulkan {
class CommandPool;
class DeviceMemory;
class Image;
class ImageView;
class Sampler;
}

namespace assets {
class Texture;

class TextureImage final {
public:

    TextureImage(const TextureImage&) = delete;
    TextureImage(TextureImage&&) = delete;
    TextureImage& operator=(const TextureImage&) = delete;
    TextureImage& operator=(TextureImage&&) = delete;

    TextureImage(vulkan::CommandPool& commandPool, const Texture& texture);
    ~TextureImage();

    const vulkan::ImageView& ImageView() const { return *imageView_; }
    const vulkan::Sampler& Sampler() const { return *sampler_; }

private:

    std::unique_ptr<vulkan::Image> image_;
    std::unique_ptr<vulkan::DeviceMemory> imageMemory_;
    std::unique_ptr<vulkan::ImageView> imageView_;
    std::unique_ptr<vulkan::Sampler> sampler_;
};

}
