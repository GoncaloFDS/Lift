#include "frame_buffer.h"
#include "depth_buffer.h"
#include "device.h"
#include "image_view.h"
#include "render_pass.h"
#include "swap_chain.h"
#include <array>

namespace vulkan {

FrameBuffer::FrameBuffer(const class ImageView& image_view, const class RenderPass& render_pass) :
    image_view_(image_view), render_pass_(render_pass) {

    std::array<VkImageView, 2> attachments = {image_view.handle(), render_pass.depthBuffer().imageView().handle()};

    VkFramebufferCreateInfo framebuffer_info = {};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = render_pass.handle();
    framebuffer_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebuffer_info.pAttachments = attachments.data();
    framebuffer_info.width = render_pass.swapChain().extent().width;
    framebuffer_info.height = render_pass.swapChain().extent().height;
    framebuffer_info.layers = 1;

    vulkanCheck(vkCreateFramebuffer(image_view_.device().handle(), &framebuffer_info, nullptr, &framebuffer_),
                "create framebuffer");
}

FrameBuffer::FrameBuffer(FrameBuffer&& other) noexcept :
    image_view_(other.image_view_), render_pass_(other.render_pass_), framebuffer_(other.framebuffer_) {
    other.framebuffer_ = nullptr;
}

FrameBuffer::~FrameBuffer() {
    if (framebuffer_ != nullptr) {
        vkDestroyFramebuffer(image_view_.device().handle(), framebuffer_, nullptr);
        framebuffer_ = nullptr;
    }
}

}  // namespace vulkan
