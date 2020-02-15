#pragma once

#include "core/utilities.h"

namespace vulkan {
class ImageView;
class RenderPass;

class FrameBuffer final {
public:
    FrameBuffer(const FrameBuffer &) = delete;
    FrameBuffer &operator=(const FrameBuffer &) = delete;
    FrameBuffer &operator=(FrameBuffer &&) = delete;

    explicit FrameBuffer(const ImageView &image_view, const RenderPass &render_pass);
    FrameBuffer(FrameBuffer &&other) noexcept;
    ~FrameBuffer();

    [[nodiscard]] VkFramebuffer handle() const { return framebuffer_; }
    [[nodiscard]] const class ImageView &imageView() const { return image_view_; }
    [[nodiscard]] const class RenderPass &renderPass() const { return render_pass_; }

private:
    const class ImageView &image_view_;
    const class RenderPass &render_pass_;
    VkFramebuffer framebuffer_ {};
};

}  // namespace vulkan
