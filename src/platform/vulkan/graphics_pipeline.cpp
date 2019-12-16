#include "graphics_pipeline.h"

#include <memory>
#include "buffer.h"
#include "descriptor_set_manager.h"
#include "descriptor_pool.h"
#include "descriptor_sets.h"
#include "device.h"
#include "pipeline_layout.h"
#include "render_pass.h"
#include "shader_module.h"
#include "swap_chain.h"
#include "assets/scene.h"
#include "assets/uniform_buffer.h"
#include "assets/vertex.h"

namespace vulkan {

GraphicsPipeline::GraphicsPipeline(const SwapChain& swap_chain,
                                   const DepthBuffer& depth_buffer,
                                   const std::vector<assets::UniformBuffer>& uniform_buffers,
                                   const assets::Scene& scene,
                                   const bool is_wire_frame) : swap_chain_(swap_chain), is_wire_frame_(is_wire_frame) {
    const auto& device = swap_chain.device();
    const auto binding_description = assets::Vertex::GetBindingDescription();
    const auto attribute_descriptions = assets::Vertex::GetAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 1;
    vertex_input_info.pVertexBindingDescriptions = &binding_description;
    vertex_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descriptions.size());
    vertex_input_info.pVertexAttributeDescriptions = attribute_descriptions.data();

    VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swap_chain.extent().width);
    viewport.height = static_cast<float>(swap_chain.extent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = swap_chain.extent();

    VkPipelineViewportStateCreateInfo viewport_state = {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = is_wire_frame ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    VkPipelineMultisampleStateCreateInfo multi_sampling = {};
    multi_sampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multi_sampling.sampleShadingEnable = VK_FALSE;
    multi_sampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multi_sampling.minSampleShading = 1.0f; // Optional
    multi_sampling.pSampleMask = nullptr; // Optional
    multi_sampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multi_sampling.alphaToOneEnable = VK_FALSE; // Optional

    VkPipelineDepthStencilStateCreateInfo depth_stencil = {};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.minDepthBounds = 0.0f; // Optional
    depth_stencil.maxDepthBounds = 1.0f; // Optional
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.front = {}; // Optional
    depth_stencil.back = {}; // Optional

    VkPipelineColorBlendAttachmentState color_blend_attachment = {};
    color_blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = VK_FALSE;
    color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
    color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

    VkPipelineColorBlendStateCreateInfo color_blending = {};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY; // Optional
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;
    color_blending.blendConstants[0] = 0.0f; // Optional
    color_blending.blendConstants[1] = 0.0f; // Optional
    color_blending.blendConstants[2] = 0.0f; // Optional
    color_blending.blendConstants[3] = 0.0f; // Optional

    // Create descriptor pool/sets.
    std::vector<DescriptorBinding> descriptor_bindings = {
        {0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
        {1, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
        {2, static_cast<uint32_t>(scene.TextureSamplers().size()), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         VK_SHADER_STAGE_FRAGMENT_BIT}
    };

    descriptor_set_manager_ = std::make_unique<DescriptorSetManager>(device, descriptor_bindings, uniform_buffers.size());

    auto& descriptor_sets = descriptor_set_manager_->descriptorSets();

    for (uint32_t i = 0; i != swap_chain.images().size(); ++i) {
        // Uniform buffer
        VkDescriptorBufferInfo uniform_buffer_info = {};
        uniform_buffer_info.buffer = uniform_buffers[i].Buffer().Handle();
        uniform_buffer_info.range = VK_WHOLE_SIZE;

        // Material buffer
        VkDescriptorBufferInfo material_buffer_info = {};
        material_buffer_info.buffer = scene.MaterialBuffer().Handle();
        material_buffer_info.range = VK_WHOLE_SIZE;

        // Image and texture samplers
        std::vector<VkDescriptorImageInfo> imageInfos(scene.TextureSamplers().size());

        for (size_t t = 0; t != imageInfos.size(); ++t) {
            auto& image_info = imageInfos[t];
            image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            image_info.imageView = scene.TextureImageViews()[t];
            image_info.sampler = scene.TextureSamplers()[t];
        }

        const std::vector<VkWriteDescriptorSet> descriptor_writes = {
            descriptor_sets.bind(i, 0, uniform_buffer_info),
            descriptor_sets.bind(i, 1, material_buffer_info),
            descriptor_sets.bind(i, 2, *imageInfos.data(), static_cast<uint32_t>(imageInfos.size()))
        };

        descriptor_sets.updateDescriptors(descriptor_writes);
    }

    // Create pipeline layout and render pass.
    pipeline_layout_ = std::make_unique<class PipelineLayout>(device, descriptor_set_manager_->descriptorSetLayout());
    render_pass_ = std::make_unique<class RenderPass>(swap_chain, depth_buffer, true, true);

    // Load shaders.
    const ShaderModule vert_shader(device, "../resources/shaders/Graphics.vert.spv");
    const ShaderModule frag_shader(device, "../resources/shaders/Graphics.frag.spv");

    VkPipelineShaderStageCreateInfo shader_stages[] = {
        vert_shader.createShaderStage(VK_SHADER_STAGE_VERTEX_BIT),
        frag_shader.createShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT)
    };

    // Create graphic pipeline
    VkGraphicsPipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multi_sampling;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = nullptr; // Optional
    pipeline_info.basePipelineHandle = nullptr; // Optional
    pipeline_info.basePipelineIndex = -1; // Optional
    pipeline_info.layout = pipeline_layout_->Handle();
    pipeline_info.renderPass = render_pass_->Handle();
    pipeline_info.subpass = 0;

    vulkanCheck(vkCreateGraphicsPipelines(device.Handle(), nullptr, 1, &pipeline_info, nullptr, &pipeline_),
                "create graphics pipeline");
}

GraphicsPipeline::~GraphicsPipeline() {
    if (pipeline_ != nullptr) {
        vkDestroyPipeline(swap_chain_.device().Handle(), pipeline_, nullptr);
        pipeline_ = nullptr;
    }

    render_pass_.reset();
    pipeline_layout_.reset();
    descriptor_set_manager_.reset();
}

VkDescriptorSet GraphicsPipeline::descriptorSet(const uint32_t index) const {
    return descriptor_set_manager_->descriptorSets().Handle(index);
}

}
