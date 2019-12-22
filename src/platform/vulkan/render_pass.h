#pragma once

#include "core/utilities.h"

namespace vulkan {
class DepthBuffer;
class SwapChain;

class RenderPass final {
public:
    RenderPass(const SwapChain& swap_chain,
               const DepthBuffer& depth_buffer,
               bool clear_color_buffer,
               bool clear_depth_buffer);
    ~RenderPass();

    [[nodiscard]] const class SwapChain& swapChain() const { return swap_chain_; }
    [[nodiscard]] const class DepthBuffer& depthBuffer() const { return depth_buffer_; }

private:

    const class SwapChain& swap_chain_;
    const class DepthBuffer& depth_buffer_;

VULKAN_HANDLE(VkRenderPass, renderPass_)
};

}
