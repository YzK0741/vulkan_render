module;

#include <vulkan/vulkan.h>

module vulkan.core.pipeline;
import vulkan.core.pipeline.spirv_parser;
import vulkan.constant_init;

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
    std::expected<vk_pipeline, std::string_view> make_pipeline(
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
        // Single-target convenience form: forward the (0 or 1)-element format list to the
        // multi-target implementation below.
        std::array<VkFormat, 1> const single_format = {color_format};
        std::span<VkFormat const> const color_formats = has_color_attachment ? std::span<VkFormat const>(single_format) : std::span<VkFormat const>{};
        return make_pipeline(device,
                             pipeline_layout,
                             color_formats,
                             depth_format,
                             vertex_shader_code,
                             fragment_shader_code,
                             msaa_level,
                             depth_test_enabled,
                             depth_bias_constant_factor,
                             depth_bias_slope_factor,
                             depth_bias_clamp,
                             true); // the forward pipelines blend (alpha is coverage; opaque alpha 1)
    }

    std::expected<vk_pipeline, std::string_view> make_pipeline( // NOLINT(*-function-cognitive-complexity)
        VkDevice device,
        VkPipelineLayout const pipeline_layout, // shared scene layout (fixed set 0 + push block); not owned
        std::span<VkFormat const> const color_formats,
        VkFormat const depth_format,
        std::span<unsigned char const> const vertex_shader_code,
        std::span<unsigned char const> const fragment_shader_code,
        VkSampleCountFlagBits const msaa_level,
        bool const depth_test_enabled,
        float const depth_bias_constant_factor,
        float const depth_bias_slope_factor,
        float const depth_bias_clamp,
        bool const color_blending) {
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
        // one blend attachment per color target: the forward pipelines blend (alpha is coverage, so
        // an opaque draw's alpha 1 reduces the blend math to the source color), while a G-buffer
        // pipeline overwrites - there alpha is metallic/roughness/flags data, and src-alpha blending
        // would mix the stored surface with the cleared target (see
        // make_color_blend_attachment_opaque)
        VkPipelineColorBlendAttachmentState const blend_attachment = color_blending ? make_color_blend_attachment() : make_color_blend_attachment_opaque();
        std::vector<VkPipelineColorBlendAttachmentState> const color_blend_attachments(color_formats.size(), blend_attachment);
        VkPipelineColorBlendStateCreateInfo const color_blend_state_create_info = make_color_blend_state(color_blend_attachments.data(), static_cast<uint32_t>(color_blend_attachments.size()));
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
        VkPipelineRenderingCreateInfo const rendering_create_info = make_rendering_create_info(color_formats.data(), static_cast<uint32_t>(color_formats.size()), depth_format);
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
