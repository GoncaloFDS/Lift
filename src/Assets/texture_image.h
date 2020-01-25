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

    TextureImage(vulkan::CommandPool& command_pool, const Texture& texture);
    ~TextureImage();

    [[nodiscard]] const vulkan::ImageView& imageView() const { return *image_view_; }
    [[nodiscard]] const vulkan::Sampler& sampler() const { return *sampler_; }

private:

    std::unique_ptr<vulkan::Image> image_;
    std::unique_ptr<vulkan::DeviceMemory> image_memory_;
    std::unique_ptr<vulkan::ImageView> image_view_;
    std::unique_ptr<vulkan::Sampler> sampler_;
};

}
