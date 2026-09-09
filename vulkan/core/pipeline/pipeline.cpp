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
    // ---- Fixed-function pipeline state, built by constexpr factories (see the call site
    //      below): each state is either fully constant or differs by one or two knobs, so the
    //      hand re-fills are replaced by a single construction. Pointer members point at
    //      caller-owned data (never at locals of the factory itself). ----
    constexpr VkPipelineShaderStageCreateInfo make_shader_stage(VkShaderModule const module, VkShaderStageFlagBits const stage) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = stage,
                .module = module,
                .pName = "main",
                .pSpecializationInfo = nullptr};
    }
    constexpr VkPipelineInputAssemblyStateCreateInfo make_input_assembly_state() noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                .primitiveRestartEnable = VK_FALSE};
    }
    constexpr VkPipelineViewportStateCreateInfo make_viewport_state() noexcept {
        // viewport/scissor are dynamic state: only the counts are static
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .viewportCount = 1,
                .pViewports = nullptr,
                .scissorCount = 1,
                .pScissors = nullptr};
    }
    constexpr VkPipelineDynamicStateCreateInfo make_dynamic_state(VkDynamicState const* states, uint32_t const count) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .dynamicStateCount = count,
                .pDynamicStates = states};
    }
    constexpr VkPipelineRasterizationStateCreateInfo make_rasterization_state(
        bool const depth_bias_enabled,
        float const depth_bias_constant_factor,
        float const depth_bias_slope_factor,
        float const depth_bias_clamp) noexcept {
        // slope-scaled depth bias for depth-writing passes (the shadow map); when the bias is
        // enabled the factors are dynamic state anyway - the static values only matter while
        // the dynamic state is never set
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_BACK_BIT,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = depth_bias_enabled ? VK_TRUE : VK_FALSE,
                .depthBiasConstantFactor = depth_bias_constant_factor,
                .depthBiasClamp = depth_bias_clamp,
                .depthBiasSlopeFactor = depth_bias_slope_factor,
                .lineWidth = 1.0f};
    }
    constexpr VkPipelineDepthStencilStateCreateInfo make_depth_stencil_state(bool const depth_test_enabled) noexcept {
        // depth test + write disabled for background passes (e.g. the skybox), which draw first
        // and must not occlude later geometry
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .depthTestEnable = depth_test_enabled ? VK_TRUE : VK_FALSE,
                .depthWriteEnable = depth_test_enabled ? VK_TRUE : VK_FALSE,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
                .front = {},
                .back = {},
                .minDepthBounds = 0.0f,
                .maxDepthBounds = 0.0f};
    }
    // Standard alpha blending, ALWAYS enabled: with src alpha == 1 (an opaque material) the
    // blend math reduces to the source color exactly, so opaque draws are pixel-identical
    // whether or not blending is on. Blended (transparent) materials simply carry alpha < 1
    // and are drawn depth-write-off in the transparent pass - no second pipeline needed.
    constexpr VkPipelineColorBlendAttachmentState make_color_blend_attachment() noexcept {
        return {.blendEnable = VK_TRUE,
                .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .alphaBlendOp = VK_BLEND_OP_ADD,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    }
    constexpr VkPipelineColorBlendStateCreateInfo make_color_blend_state(bool const has_color_attachment, VkPipelineColorBlendAttachmentState const* attachment) noexcept {
        // depth-only pipelines (e.g. the shadow pass) have no color attachment: no blend state
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .logicOpEnable = VK_FALSE,
                .logicOp = VK_LOGIC_OP_COPY,
                .attachmentCount = has_color_attachment ? 1u : 0u,
                .pAttachments = has_color_attachment ? attachment : nullptr,
                .blendConstants = {1.0f, 1.0f, 1.0f, 1.0f}};
    }
    constexpr VkPipelineMultisampleStateCreateInfo make_multisample_state(VkSampleCountFlagBits const samples) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .rasterizationSamples = samples,
                .sampleShadingEnable = VK_FALSE,
                .minSampleShading = 0.0f,
                .pSampleMask = nullptr,
                .alphaToCoverageEnable = VK_FALSE,
                .alphaToOneEnable = VK_FALSE};
    }
    // @param color_format_ptr pointer to the caller's color format: the returned info holds
    //        that pointer, so it must outlive the struct (it does - make_pipeline's parameter)
    constexpr VkPipelineRenderingCreateInfo make_rendering_create_info(bool const has_color_attachment, VkFormat const* color_format_ptr, VkFormat const depth_format) noexcept {
        return {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                .pNext = nullptr,
                .viewMask = 0,
                .colorAttachmentCount = has_color_attachment ? 1u : 0u,
                .pColorAttachmentFormats = has_color_attachment ? color_format_ptr : nullptr,
                .depthAttachmentFormat = depth_format,
                .stencilAttachmentFormat = VK_FORMAT_UNDEFINED};
    }

    std::expected<vk_pipeline, std::string_view> make_pipeline( // NOLINT(*-function-cognitive-complexity)
        VkDevice device,
        VkPipelineLayout const pipeline_layout, // shared scene layout (fixed set 0 + push block); not owned
        VkFormat const color_format,
        VkFormat const depth_format,
        std::span<unsigned char const> const vertex_shader_code,
        std::span<unsigned char const> const fragment_shader_code,
        VkSampleCountFlagBits const msaa_level,
        bool const depth_test_enabled,
        bool const has_color_attachment,
        float const depth_bias_constant_factor,
        float const depth_bias_slope_factor,
        float const depth_bias_clamp) {
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

        // ---- 3. Fixed-function pipeline state (constexpr factories, see above) ----
        std::array<VkPipelineShaderStageCreateInfo, 2> shader_stage_create_infos = {
            make_shader_stage(**vertex_shader_module, VK_SHADER_STAGE_VERTEX_BIT),
            make_shader_stage(**fragment_shader_module, VK_SHADER_STAGE_FRAGMENT_BIT),
        };

        // double-sided materials need per-draw cull control (core dynamic state since Vulkan 1.3);
        // transparent (alphaMode BLEND) leaves disable depth writes per draw, also a 1.3 core
        // dynamic state, so one pipeline serves both opaque and blended draws
        std::array<VkDynamicState, 6> dynamic_states = {};
        uint32_t dynamic_state_count = 4;
        dynamic_states[0] = VK_DYNAMIC_STATE_VIEWPORT;
        dynamic_states[1] = VK_DYNAMIC_STATE_SCISSOR;
        dynamic_states[2] = VK_DYNAMIC_STATE_CULL_MODE;
        dynamic_states[3] = VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE;
        // depth-bias pipelines (the shadow pass) take the three bias factors as dynamic state,
        // so the values can be tuned live (e.g. from the debug gui) without recreating the
        // pipeline; depthBiasEnable itself stays static below
        bool const depth_bias_enabled = depth_bias_constant_factor != 0.0f || depth_bias_slope_factor != 0.0f || depth_bias_clamp != 0.0f;
        if (depth_bias_enabled) {
            dynamic_states[dynamic_state_count++] = VK_DYNAMIC_STATE_DEPTH_BIAS;
        }

        VkPipelineInputAssemblyStateCreateInfo const input_assembly_state_create_info = make_input_assembly_state();
        VkPipelineViewportStateCreateInfo const viewport_state_create_info = make_viewport_state();
        VkPipelineDynamicStateCreateInfo const dynamic_state_create_info = make_dynamic_state(dynamic_states.data(), dynamic_state_count);
        VkPipelineRasterizationStateCreateInfo const rasterization_state_create_info = make_rasterization_state(depth_bias_enabled, depth_bias_constant_factor, depth_bias_slope_factor, depth_bias_clamp);
        VkPipelineDepthStencilStateCreateInfo const depth_stencil_state_create_info = make_depth_stencil_state(depth_test_enabled);
        VkPipelineColorBlendAttachmentState const color_blend_attachment_state = make_color_blend_attachment();
        VkPipelineColorBlendStateCreateInfo const color_blend_state_create_info = make_color_blend_state(has_color_attachment, &color_blend_attachment_state);
        VkPipelineMultisampleStateCreateInfo const multisample_state_create_info = make_multisample_state(msaa_level);

        // ---- 4. Pipeline layout: the shared scene layout (passed in) already carries the
        //         agreed flat descriptor set 0 and the fixed push constant block; nothing to
        //         parse from SPIR-V for the indexed layout (see core::init_scene_layouts) ----
        resource_guard guard;
        guard.device = device;

        // ---- 5. graphics pipeline ----
        // Dynamic rendering (Vulkan 1.3 core, the only path the engine uses): attachments are
        // declared through VkPipelineRenderingCreateInfo in the pNext chain instead of a render
        // pass + subpass. Depth-only pipelines (has_color_attachment == false, e.g. the shadow
        // pass) declare no color attachment format.
        VkPipelineRenderingCreateInfo const rendering_create_info = make_rendering_create_info(has_color_attachment, &color_format, depth_format);
        VkGraphicsPipelineCreateInfo pipeline_create_info = {};
        pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_create_info.pNext = &rendering_create_info;
        pipeline_create_info.renderPass = VK_NULL_HANDLE; // dynamic rendering: no render pass
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
