module;

#include <vulkan/vulkan.h>

module vulkan.core.pipeline;
import vulkan.core.pipeline.spirv_parser;

namespace {
    // Collects the Vulkan objects created during pipeline creation; the destructor frees the
    // pipeline if any later step fails; on full success, release() surrenders ownership to vk_pipeline.
    struct resource_guard {
        VkDevice device = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        void release() noexcept {
            this->device = VK_NULL_HANDLE;
            this->pipeline = VK_NULL_HANDLE;
        }

        ~resource_guard() {
            if (this->device == VK_NULL_HANDLE) {
                return;
            }
            if (this->pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(this->device, this->pipeline, nullptr);
            }
        }
    };
} // namespace

namespace vulkan {
    std::expected<vk_pipeline, std::string_view> make_pipeline( // NOLINT(*-function-cognitive-complexity)
        VkDevice device,
        VkPipelineLayout const pipeline_layout, // shared scene layout (fixed set 0 + push block); not owned
        VkRenderPass const render_pass,         // NOLINT(*-misplaced-const) ; VK_NULL_HANDLE selects dynamic rendering
        VkFormat const color_format,
        VkFormat const depth_format,
        std::span<unsigned char const> const vertex_shader_code,
        std::span<unsigned char const> const fragment_shader_code,
        VkSampleCountFlagBits const msaa_level,
        bool const depth_test_enabled) {
        using fail = std::unexpected<std::string_view>;

        // ---- 1. Parse the vertex stage interface, filter builtins, build vertex input ----
        auto vertex_interface_expected = pipeline::parse_shader_stage_interface(vertex_shader_code, VK_SHADER_STAGE_VERTEX_BIT);
        if (!vertex_interface_expected) {
            return fail(vertex_interface_expected.error());
        }
        auto const vertex_interface = std::move(vertex_interface_expected).value();

        std::vector<VkVertexInputAttributeDescription> attribute_descriptions;
        attribute_descriptions.reserve(vertex_interface.inputs.size());
        uint32_t stride = 0;
        for (auto const& variable : vertex_interface.inputs) {
            if (variable.is_builtin || variable.format == VK_FORMAT_UNDEFINED) {
                continue;
            }
            attribute_descriptions.push_back(VkVertexInputAttributeDescription{
                .location = variable.location,
                .binding = 0,
                .format = variable.format,
                .offset = stride,
            });
            stride += pipeline::format_size(variable.format);
        }

        VkVertexInputBindingDescription vertex_input_binding = {};
        vertex_input_binding.binding = 0;
        vertex_input_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        vertex_input_binding.stride = stride;

        VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {};
        vertex_input_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_state_create_info.vertexBindingDescriptionCount = attribute_descriptions.empty() ? 0u : 1u;
        vertex_input_state_create_info.pVertexBindingDescriptions = attribute_descriptions.empty() ? nullptr : &vertex_input_binding;
        vertex_input_state_create_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descriptions.size());
        vertex_input_state_create_info.pVertexAttributeDescriptions = attribute_descriptions.empty() ? nullptr : attribute_descriptions.data();

        // ---- 2. Create shader modules (errors only on failure) ----
        auto vertex_shader_module = make_shader_module(vertex_shader_code, device);
        if (!vertex_shader_module) {
            return fail("failed to create vertex shader module");
        }

        auto fragment_shader_module = make_shader_module(fragment_shader_code, device);
        if (!fragment_shader_module) {
            return fail("failed to create fragment shader module");
        }

        // ---- 3. Fixed-function pipeline state ----
        VkPipelineShaderStageCreateInfo vertex_stage = {};
        vertex_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertex_stage.module = **vertex_shader_module;
        vertex_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertex_stage.pName = "main";
        vertex_stage.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo fragment_stage = {};
        fragment_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragment_stage.module = **fragment_shader_module;
        fragment_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragment_stage.pName = "main";
        fragment_stage.pSpecializationInfo = nullptr;

        std::array<VkPipelineShaderStageCreateInfo, 2> shader_stage_create_infos = {vertex_stage, fragment_stage};

        VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {};
        input_assembly_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly_state_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly_state_create_info.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewport_state_create_info = {};
        viewport_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state_create_info.viewportCount = 1;
        viewport_state_create_info.scissorCount = 1;
        viewport_state_create_info.pScissors = nullptr;
        viewport_state_create_info.pViewports = nullptr;

        std::array<VkDynamicState, 3> dynamic_states = {};
        dynamic_states[0] = VK_DYNAMIC_STATE_VIEWPORT;
        dynamic_states[1] = VK_DYNAMIC_STATE_SCISSOR;
        // double-sided materials need per-draw cull control (core dynamic state since Vulkan 1.3)
        dynamic_states[2] = VK_DYNAMIC_STATE_CULL_MODE;

        VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {};
        dynamic_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state_create_info.dynamicStateCount = dynamic_states.size();
        dynamic_state_create_info.pDynamicStates = dynamic_states.data();

        VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {};
        rasterization_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization_state_create_info.depthClampEnable = VK_FALSE;
        rasterization_state_create_info.rasterizerDiscardEnable = VK_FALSE;
        rasterization_state_create_info.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization_state_create_info.lineWidth = 1.0f;
        rasterization_state_create_info.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterization_state_create_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization_state_create_info.depthBiasEnable = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info = {};
        depth_stencil_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        // depth test + write disabled for background passes (e.g. the skybox), which draw first
        // and must not occlude later geometry
        depth_stencil_state_create_info.depthTestEnable = depth_test_enabled ? VK_TRUE : VK_FALSE;
        depth_stencil_state_create_info.depthWriteEnable = depth_test_enabled ? VK_TRUE : VK_FALSE;
        depth_stencil_state_create_info.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState color_blend_attachment_state = {};
        color_blend_attachment_state.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        color_blend_attachment_state.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {};
        color_blend_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend_state_create_info.logicOpEnable = VK_FALSE;
        color_blend_state_create_info.logicOp = VK_LOGIC_OP_COPY;
        color_blend_state_create_info.attachmentCount = 1;
        color_blend_state_create_info.blendConstants[0] = 1.0f;
        color_blend_state_create_info.blendConstants[1] = 1.0f;
        color_blend_state_create_info.blendConstants[2] = 1.0f;
        color_blend_state_create_info.blendConstants[3] = 1.0f;
        color_blend_state_create_info.pAttachments = &color_blend_attachment_state;

        VkPipelineMultisampleStateCreateInfo multisample_state_create_info = {};
        multisample_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample_state_create_info.sampleShadingEnable = VK_FALSE;
        multisample_state_create_info.rasterizationSamples = msaa_level;
        multisample_state_create_info.pSampleMask = nullptr;
        multisample_state_create_info.alphaToCoverageEnable = VK_FALSE;
        multisample_state_create_info.alphaToOneEnable = VK_FALSE;

        // ---- 4. Pipeline layout: the shared scene layout (passed in) already carries the
        //         agreed flat descriptor set 0 and the fixed push constant block; nothing to
        //         parse from SPIR-V for the indexed layout (see core::init_scene_layouts) ----
        resource_guard guard;
        guard.device = device;

        // ---- 5. graphics pipeline ----
        // With dynamic rendering (render_pass == VK_NULL_HANDLE) the attachments are described
        // by VkPipelineRenderingCreateInfo in the pNext chain instead of a render pass + subpass
        VkPipelineRenderingCreateInfo rendering_create_info = {};
        if (render_pass == VK_NULL_HANDLE) {
            rendering_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            rendering_create_info.colorAttachmentCount = 1;
            rendering_create_info.pColorAttachmentFormats = &color_format;
            rendering_create_info.depthAttachmentFormat = depth_format;
        }
        VkGraphicsPipelineCreateInfo pipeline_create_info = {};
        pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_create_info.pNext = render_pass == VK_NULL_HANDLE ? &rendering_create_info : nullptr;
        pipeline_create_info.renderPass = render_pass;
        pipeline_create_info.pInputAssemblyState = &input_assembly_state_create_info;
        pipeline_create_info.pViewportState = &viewport_state_create_info;
        pipeline_create_info.pDepthStencilState = &depth_stencil_state_create_info;
        pipeline_create_info.pColorBlendState = &color_blend_state_create_info;
        pipeline_create_info.pVertexInputState = &vertex_input_state_create_info;
        pipeline_create_info.layout = pipeline_layout;
        pipeline_create_info.pRasterizationState = &rasterization_state_create_info;
        pipeline_create_info.pMultisampleState = &multisample_state_create_info;
        pipeline_create_info.pDynamicState = &dynamic_state_create_info;
        pipeline_create_info.stageCount = 2;
        pipeline_create_info.pStages = shader_stage_create_infos.data();
        pipeline_create_info.subpass = 0;
        pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_create_info, nullptr, &guard.pipeline) != VK_SUCCESS) {
            return fail("failed to create graphics pipeline");
        }

        // ---- 8. Success: transfer ownership to vk_pipeline; guard no longer cleans up ----
        //      (the pipeline layout is shared and owned by core, not by the pipeline)
        vk_pipeline result(guard.pipeline, pipeline_layout, device);
        guard.release();
        return result;
    }
} // namespace vulkan
