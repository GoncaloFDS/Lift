#pragma once

#include "core/utilities.h"
#include "image_view.h"
#include <memory>
#include <vector>

class vkDescriptorImageInfo;

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
    GraphicsPipeline(const SwapChain& swap_chain, const DepthBuffer& depth_buffer);
    ~GraphicsPipeline();

    [[nodiscard]] VkPipeline handle() const { return pipeline_; }

    [[nodiscard]] VkDescriptorSet descriptorSet(uint32_t index) const;
    [[nodiscard]] const PipelineLayout& pipelineLayout() const { return *pipeline_layout_; }
    [[nodiscard]] const RenderPass& renderPass() const { return *render_pass_; }

    void updateOutputImage(const vulkan::ImageView& output_image, const uint32_t image_index);

private:
    VkPipeline pipeline_ {};
    const SwapChain& swap_chain_;

    std::unique_ptr<class DescriptorSetManager> descriptor_set_manager_;
    std::unique_ptr<class PipelineLayout> pipeline_layout_;
    std::unique_ptr<class RenderPass> render_pass_;
    std::unique_ptr<class Sampler> sampler_;
};

}  // namespace vulkan
