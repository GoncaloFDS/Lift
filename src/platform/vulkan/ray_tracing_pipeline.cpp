#include "ray_tracing_pipeline.h"

#include <memory>
#include "device_procedures.h"
#include "tlas.h"
#include "assets/scene.h"
#include "assets/uniform_buffer.h"
#include "platform/vulkan/buffer.h"
#include "platform/vulkan/device.h"
#include "platform/vulkan/descriptor_binding.h"
#include "platform/vulkan/descriptor_set_manager.h"
#include "platform/vulkan/descriptor_sets.h"
#include "platform/vulkan/image_view.h"
#include "platform/vulkan/pipeline_layout.h"
#include "platform/vulkan/shader_module.h"
#include "platform/vulkan/swap_chain.h"

namespace vulkan {

RayTracingPipeline::RayTracingPipeline(
    const DeviceProcedures& device_procedures,
    const SwapChain& swap_chain,
    const TopLevelAccelerationStructure& acceleration_structure,
    const ImageView& accumulation_image_view,
    const ImageView& output_image_view,
    const std::vector<assets::UniformBuffer>& uniform_buffers,
    const assets::Scene& scene) :
    swap_chain_(swap_chain) {
    // Create descriptor pool/sets.
    const auto& device = swap_chain.device();
    const std::vector<DescriptorBinding> descriptorBindings =
        {
            // Top level acceleration structure.
            {0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, VK_SHADER_STAGE_RAYGEN_BIT_NV},

            // Image accumulation & output
            {1, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_NV},
            {2, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_NV},

            // Camera information & co
            {3, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_MISS_BIT_NV},

            // Vertex buffer, Index buffer, Material buffer, Offset buffer
            {4, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV},
            {5, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV},
            {6, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV},
            {7, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV},

            // Textures and image samplers
            {8, static_cast<uint32_t>(scene.textureSamplers().size()), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV},

            // The Procedural buffer.
            {9, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV | VK_SHADER_STAGE_INTERSECTION_BIT_NV}
        };

    descriptor_set_manager_ = std::make_unique<DescriptorSetManager>(device, descriptorBindings, uniform_buffers.size());

    auto& descriptorSets = descriptor_set_manager_->descriptorSets();

    for (uint32_t i = 0; i != swap_chain.images().size(); ++i) {
        // Top level acceleration structure.
        const auto accelerationStructureHandle = acceleration_structure.handle();
        VkWriteDescriptorSetAccelerationStructureNV structureInfo = {};
        structureInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
        structureInfo.pNext = nullptr;
        structureInfo.accelerationStructureCount = 1;
        structureInfo.pAccelerationStructures = &accelerationStructureHandle;

        // Accumulation image
        VkDescriptorImageInfo accumulationImageInfo = {};
        accumulationImageInfo.imageView = accumulation_image_view.handle();
        accumulationImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Output image
        VkDescriptorImageInfo outputImageInfo = {};
        outputImageInfo.imageView = output_image_view.handle();
        outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Uniform buffer
        VkDescriptorBufferInfo uniformBufferInfo = {};
        uniformBufferInfo.buffer = uniform_buffers[i].buffer().handle();
        uniformBufferInfo.range = VK_WHOLE_SIZE;

        // Vertex buffer
        VkDescriptorBufferInfo vertexBufferInfo = {};
        vertexBufferInfo.buffer = scene.vertexBuffer().handle();
        vertexBufferInfo.range = VK_WHOLE_SIZE;

        // Index buffer
        VkDescriptorBufferInfo indexBufferInfo = {};
        indexBufferInfo.buffer = scene.indexBuffer().handle();
        indexBufferInfo.range = VK_WHOLE_SIZE;

        // Material buffer
        VkDescriptorBufferInfo materialBufferInfo = {};
        materialBufferInfo.buffer = scene.materialBuffer().handle();
        materialBufferInfo.range = VK_WHOLE_SIZE;

        // Offsets buffer
        VkDescriptorBufferInfo offsetsBufferInfo = {};
        offsetsBufferInfo.buffer = scene.offsetsBuffer().handle();
        offsetsBufferInfo.range = VK_WHOLE_SIZE;

        // Image and texture samplers.
        std::vector<VkDescriptorImageInfo> imageInfos(scene.textureSamplers().size());

        for (size_t t = 0; t != imageInfos.size(); ++t) {
            auto& imageInfo = imageInfos[t];
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = scene.textureImageViews()[t];
            imageInfo.sampler = scene.textureSamplers()[t];
        }

        std::vector<VkWriteDescriptorSet> descriptorWrites =
            {
                descriptorSets.bind(i, 0, structureInfo),
                descriptorSets.bind(i, 1, accumulationImageInfo),
                descriptorSets.bind(i, 2, outputImageInfo),
                descriptorSets.bind(i, 3, uniformBufferInfo),
                descriptorSets.bind(i, 4, vertexBufferInfo),
                descriptorSets.bind(i, 5, indexBufferInfo),
                descriptorSets.bind(i, 6, materialBufferInfo),
                descriptorSets.bind(i, 7, offsetsBufferInfo),
                descriptorSets.bind(i, 8, *imageInfos.data(), static_cast<uint32_t>(imageInfos.size()))
            };

        // Procedural buffer (optional)
        VkDescriptorBufferInfo proceduralBufferInfo = {};

        if (scene.hasProcedurals()) {
            proceduralBufferInfo.buffer = scene.proceduralBuffer().handle();
            proceduralBufferInfo.range = VK_WHOLE_SIZE;

            descriptorWrites.push_back(descriptorSets.bind(i, 9, proceduralBufferInfo));
        }

        descriptorSets.updateDescriptors(descriptorWrites);
    }

    pipeline_layout_ = std::make_unique<class PipelineLayout>(device, descriptor_set_manager_->descriptorSetLayout());

    // Load shaders.
    const ShaderModule rayGenShader(device, "../resources/shaders/RayTracing.rgen.spv");
    const ShaderModule missShader(device, "../resources/shaders/RayTracing.rmiss.spv");
    const ShaderModule closestHitShader(device, "../resources/shaders/RayTracing.rchit.spv");
    const ShaderModule proceduralClosestHitShader(device, "../resources/shaders/RayTracing.Procedural.rchit.spv");
    const ShaderModule proceduralIntersectionShader(device, "../resources/shaders/RayTracing.Procedural.rint.spv");

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages =
        {
            rayGenShader.createShaderStage(VK_SHADER_STAGE_RAYGEN_BIT_NV),
            missShader.createShaderStage(VK_SHADER_STAGE_MISS_BIT_NV),
            closestHitShader.createShaderStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV),
            proceduralClosestHitShader.createShaderStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV),
            proceduralIntersectionShader.createShaderStage(VK_SHADER_STAGE_INTERSECTION_BIT_NV)
        };

    // Shader groups
    VkRayTracingShaderGroupCreateInfoNV rayGenGroupInfo = {};
    rayGenGroupInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
    rayGenGroupInfo.pNext = nullptr;
    rayGenGroupInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
    rayGenGroupInfo.generalShader = 0;
    rayGenGroupInfo.closestHitShader = VK_SHADER_UNUSED_NV;
    rayGenGroupInfo.anyHitShader = VK_SHADER_UNUSED_NV;
    rayGenGroupInfo.intersectionShader = VK_SHADER_UNUSED_NV;
    ray_gen_index_ = 0;

    VkRayTracingShaderGroupCreateInfoNV missGroupInfo = {};
    missGroupInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
    missGroupInfo.pNext = nullptr;
    missGroupInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
    missGroupInfo.generalShader = 1;
    missGroupInfo.closestHitShader = VK_SHADER_UNUSED_NV;
    missGroupInfo.anyHitShader = VK_SHADER_UNUSED_NV;
    missGroupInfo.intersectionShader = VK_SHADER_UNUSED_NV;
    miss_index_ = 1;

    VkRayTracingShaderGroupCreateInfoNV triangleHitGroupInfo = {};
    triangleHitGroupInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
    triangleHitGroupInfo.pNext = nullptr;
    triangleHitGroupInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV;
    triangleHitGroupInfo.generalShader = VK_SHADER_UNUSED_NV;
    triangleHitGroupInfo.closestHitShader = 2;
    triangleHitGroupInfo.anyHitShader = VK_SHADER_UNUSED_NV;
    triangleHitGroupInfo.intersectionShader = VK_SHADER_UNUSED_NV;
    triangle_hit_group_index_ = 2;

    VkRayTracingShaderGroupCreateInfoNV proceduralHitGroupInfo = {};
    proceduralHitGroupInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
    proceduralHitGroupInfo.pNext = nullptr;
    proceduralHitGroupInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_NV;
    proceduralHitGroupInfo.generalShader = VK_SHADER_UNUSED_NV;
    proceduralHitGroupInfo.closestHitShader = 3;
    proceduralHitGroupInfo.anyHitShader = VK_SHADER_UNUSED_NV;
    proceduralHitGroupInfo.intersectionShader = 4;
    procedural_hit_group_index_ = 3;

    std::vector<VkRayTracingShaderGroupCreateInfoNV> groups =
        {
            rayGenGroupInfo,
            missGroupInfo,
            triangleHitGroupInfo,
            proceduralHitGroupInfo,
        };

    // Create graphic pipeline
    VkRayTracingPipelineCreateInfoNV pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV;
    pipelineInfo.pNext = nullptr;
    pipelineInfo.flags = 0;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.groupCount = static_cast<uint32_t>(groups.size());
    pipelineInfo.pGroups = groups.data();
    pipelineInfo.maxRecursionDepth = 1;
    pipelineInfo.layout = pipeline_layout_->handle();
    pipelineInfo.basePipelineHandle = nullptr;
    pipelineInfo.basePipelineIndex = 0;

    vulkanCheck(device_procedures.vkCreateRayTracingPipelinesNV(device.handle(),
                                                                nullptr,
                                                                1,
                                                                &pipelineInfo,
                                                                nullptr,
                                                                &pipeline_),
                "create ray tracing pipeline");
}

RayTracingPipeline::~RayTracingPipeline() {
    if (pipeline_ != nullptr) {
        vkDestroyPipeline(swap_chain_.device().handle(), pipeline_, nullptr);
        pipeline_ = nullptr;
    }

    pipeline_layout_.reset();
    descriptor_set_manager_.reset();
}

VkDescriptorSet RayTracingPipeline::descriptorSet(uint32_t index) const {
    return descriptor_set_manager_->descriptorSets().Handle(index);
}

}
