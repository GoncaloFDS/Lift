#pragma once

#include "buffer.h"
#include "core/utilities.h"
#include <algorithm_list.h>
#include <assets/lights.h>
#include <glm/vec4.hpp>
#include <memory>
#include <vector>

namespace assets {
class Scene;
class UniformBuffer;
}  // namespace assets

namespace vulkan {
class DescriptorSetManager;
class ImageView;
class PipelineLayout;
class SwapChain;
}  // namespace vulkan

namespace vulkan {
class DeviceProcedures;
class TopLevelAccelerationStructure;

class RayTracingPipeline final {
public:
    RayTracingPipeline(const DeviceProcedures& device_procedures,
                       const SwapChain& swap_chain,
                       const TopLevelAccelerationStructure& acceleration_structure,
                       const ImageView& output_image_view,
                       const std::vector<assets::UniformBuffer>& uniform_buffers,
                       const assets::Scene& scene,
                       Algorithm algorithm);
    ~RayTracingPipeline();

    [[nodiscard]] VkPipeline handle() const { return pipeline_; }

    [[nodiscard]] uint32_t rayGenShaderIndex() const { return ray_gen_index_; }
    [[nodiscard]] uint32_t missShaderIndex() const { return miss_index_; }
    [[nodiscard]] uint32_t shadowMissShaderIndex() const { return shadow_miss_index_; }
    [[nodiscard]] uint32_t triangleHitGroupIndex() const { return triangle_hit_group_index_; }
    [[nodiscard]] uint32_t proceduralHitGroupIndex() const { return procedural_hit_group_index_; }

    [[nodiscard]] VkDescriptorSet descriptorSet(uint32_t index) const;
    [[nodiscard]] const class PipelineLayout& pipelineLayout() const { return *pipeline_layout_; }

private:
    VkPipeline pipeline_ {};

    const SwapChain& swap_chain_;

    std::unique_ptr<DescriptorSetManager> descriptor_set_manager_;
    std::unique_ptr<class PipelineLayout> pipeline_layout_;

    uint32_t ray_gen_index_;
    uint32_t miss_index_;
    uint32_t shadow_miss_index_;
    uint32_t triangle_hit_group_index_;
    uint32_t procedural_hit_group_index_;

    Algorithm algorithm_;

    LightPathNode light_paths_[500];
    std::unique_ptr<vulkan::Buffer> light_paths_buffer_;
};

}  // namespace vulkan
