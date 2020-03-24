#pragma once

#include "core/utilities.h"
#include <memory>
#include <vector>

namespace assets {
class Scene;
class UniformBuffer;
}  // namespace assets

namespace vulkan {
class DepthBuffer;
class PipelineLayout;
class RenderPass;
class SwapChain;

class GraphicsPipeline final {
public:
    GraphicsPipeline(const SwapChain& swap_chain,
                     const DepthBuffer& depth_buffer,
                     const std::vector<assets::UniformBuffer>& uniform_buffers,
                     const assets::Scene& scene,
                     bool is_wire_frame);
    ~GraphicsPipeline();

    [[nodiscard]] VkPipeline handle() const { return pipeline_; }

    [[nodiscard]] VkDescriptorSet descriptorSet(uint32_t index) const;
    [[nodiscard]] bool isWireFrame() const { return is_wire_frame_; }
    [[nodiscard]] const PipelineLayout& pipelineLayout() const { return *pipeline_layout_; }
    [[nodiscard]] const RenderPass& renderPass() const { return *render_pass_; }

private:
    VkPipeline pipeline_ {};
    const SwapChain& swap_chain_;
    const bool is_wire_frame_;

    std::unique_ptr<class DescriptorSetManager> descriptor_set_manager_;
    std::unique_ptr<class PipelineLayout> pipeline_layout_;
    std::unique_ptr<class RenderPass> render_pass_;
};

}  // namespace vulkan
