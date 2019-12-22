#pragma once

#include "core/utilities.h"
#include <memory>
#include <vector>

namespace assets {
class Scene;
class UniformBuffer;
}

namespace vulkan {
class DescriptorSetManager;
class ImageView;
class PipelineLayout;
class SwapChain;
}

namespace vulkan::ray_tracing {
class DeviceProcedures;
class TopLevelAccelerationStructure;

class RayTracingPipeline final {
public:
    RayTracingPipeline(const DeviceProcedures& device_procedures,
                       const SwapChain& swap_chain,
                       const TopLevelAccelerationStructure& acceleration_structure,
                       const ImageView& accumulation_image_view,
                       const ImageView& output_image_view,
                       const std::vector<assets::UniformBuffer>& uniform_buffers,
                       const assets::Scene& scene);
    ~RayTracingPipeline();

    [[nodiscard]] uint32_t rayGenShaderIndex() const { return ray_gen_index_; }
    [[nodiscard]] uint32_t missShaderIndex() const { return miss_index_; }
    [[nodiscard]] uint32_t triangleHitGroupIndex() const { return triangle_hit_group_index_; }
    [[nodiscard]] uint32_t proceduralHitGroupIndex() const { return procedural_hit_group_index_; }

    [[nodiscard]] VkDescriptorSet descriptorSet(uint32_t index) const;
    [[nodiscard]] const class PipelineLayout& pipelineLayout() const { return *pipeline_layout_; }

private:

    const SwapChain& swap_chain_;

VULKAN_HANDLE(VkPipeline, pipeline_)

    std::unique_ptr<DescriptorSetManager> descriptor_set_manager_;
    std::unique_ptr<class PipelineLayout> pipeline_layout_;

    uint32_t ray_gen_index_;
    uint32_t miss_index_;
    uint32_t triangle_hit_group_index_;
    uint32_t procedural_hit_group_index_;
};

}
