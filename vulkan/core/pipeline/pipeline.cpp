module;

#include <vulkan/vulkan.h>

module vulkan.core.pipeline;
import vulkan.core.pipeline.spirv_parser;

namespace {
    constexpr uint32_t to_multiple_of_4(const uint32_t value) noexcept {
        return ((value + 3) / 4) * 4;
    }

    // Collects the Vulkan objects created during pipeline creation; the destructor frees them all
    // if any step fails; on full success, release() surrenders ownership (handed over to vk_pipeline).
    struct resource_guard {
        VkDevice device = VK_NULL_HANDLE;
        std::vector<VkDescriptorSetLayout> set_layouts = {};
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        void add_set_layout(const VkDescriptorSetLayout layout) noexcept {
            this->set_layouts.push_back(layout);
        }

        void release() noexcept {
            this->device = VK_NULL_HANDLE;
            this->set_layouts.clear();
            this->pipeline_layout = VK_NULL_HANDLE;
            this->pipeline = VK_NULL_HANDLE;
        }

        ~resource_guard() {
            if (this->device == VK_NULL_HANDLE) {
                return;
            }
            if (this->pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(this->device, this->pipeline, nullptr);
            }
            if (this->pipeline_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(this->device, this->pipeline_layout, nullptr);
            }
            for (const auto& layout : this->set_layouts) {
                if (layout != VK_NULL_HANDLE) {
                    vkDestroyDescriptorSetLayout(this->device, layout, nullptr);
                }
            }
        }
    };

    // Merge the descriptor layouts parsed from the vertex and fragment stages by set/binding,
    // taking the union of stageFlags when both stages share a binding.
    std::map<uint32_t, std::map<uint32_t, VkDescriptorSetLayoutBinding>> merge_descriptor_layouts(
        const std::vector<vulkan::pipeline::descriptor_set_layout_data>& vertex_layouts,
        const std::vector<vulkan::pipeline::descriptor_set_layout_data>& fragment_layouts) {
        std::map<uint32_t, std::map<uint32_t, VkDescriptorSetLayoutBinding>> merged;

        const auto merge_one = [&merged](const std::vector<vulkan::pipeline::descriptor_set_layout_data>& layouts) {
            for (const auto& set_data : layouts) {
                auto& bindings_map = merged[set_data.set_number];
                for (const auto& binding : set_data.bindings) {
                    auto [it, inserted] = bindings_map.try_emplace(binding.binding, binding);
                    if (!inserted) {
                        it->second.stageFlags |= binding.stageFlags;
                    }
                }
            }
        };

        merge_one(vertex_layouts);
        merge_one(fragment_layouts);
        return merged;
    }
} // namespace

namespace vulkan {
    std::expected<vk_pipeline, std::string_view> make_pipeline( // NOLINT(*-function-cognitive-complexity)
        VkDevice device,
        const VkRenderPass render_pass, // NOLINT(*-misplaced-const) ; VK_NULL_HANDLE selects dynamic rendering
        const VkFormat color_format,
        const VkFormat depth_format,
        const std::span<const unsigned char> vertex_shader_code,
        const std::span<const unsigned char> fragment_shader_code,
        const VkSampleCountFlagBits msaa_level) {
        using fail = std::unexpected<std::string_view>;

        // ---- 1. Parse the vertex stage interface, filter builtins, build vertex input ----
        auto vertex_interface_expected = pipeline::parse_shader_stage_interface(vertex_shader_code, VK_SHADER_STAGE_VERTEX_BIT);
        if (!vertex_interface_expected) {
            return fail(vertex_interface_expected.error());
        }
        const auto vertex_interface = std::move(vertex_interface_expected).value();

        std::vector<VkVertexInputAttributeDescription> attribute_descriptions;
        attribute_descriptions.reserve(vertex_interface.inputs.size());
        uint32_t stride = 0;
        for (const auto& variable : vertex_interface.inputs) {
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

        std::array<VkDynamicState, 2> dynamic_states = {};
        dynamic_states[0] = VK_DYNAMIC_STATE_VIEWPORT;
        dynamic_states[1] = VK_DYNAMIC_STATE_SCISSOR;

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
        depth_stencil_state_create_info.depthTestEnable = VK_TRUE;
        depth_stencil_state_create_info.depthWriteEnable = VK_TRUE;
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

        // ---- 4. Push constants: one block each for vertex/fragment, 4-byte aligned, total <= 256 ----
        auto vertex_push_constant_expected = pipeline::parse_push_constant_layout(vertex_shader_code);
        if (!vertex_push_constant_expected) {
            return fail("can't parse vertex push constant");
        }
        const auto vertex_push_constant = std::move(vertex_push_constant_expected).value();

        auto fragment_push_constant_expected = pipeline::parse_push_constant_layout(fragment_shader_code);
        if (!fragment_push_constant_expected) {
            return fail("can't parse fragment push constant");
        }
        const auto fragment_push_constant = std::move(fragment_push_constant_expected).value();

        std::vector<VkPushConstantRange> push_constant_ranges;
        push_constant_ranges.reserve(2);

        // Each stage's push constant block is relative to offset 0 (shader view),
        // so blocks that multiple stages declare over the same offset range must merge into one
        // range (union of stageFlags), otherwise the layout is illegal because a shader block
        // falls outside its stage's range (VUID-VkGraphicsPipelineCreateInfo-layout-10069).
        const auto add_push_constant_range = [&push_constant_ranges](const VkShaderStageFlags stage_flags, const uint32_t block_offset, const uint32_t block_size) -> bool {
            if (block_size == 0) {
                return true;
            }
            const uint32_t size = to_multiple_of_4(block_size);
            const uint32_t end = block_offset + size;

            for (auto& range : push_constant_ranges) {
                const uint32_t range_end = range.offset + range.size;
                if (block_offset < range_end && range.offset < end) {
                    const uint32_t merged_offset = std::min(range.offset, block_offset);
                    const uint32_t merged_end = std::max(range_end, end);
                    if (merged_end - merged_offset > 256) {
                        return false;
                    }
                    range.offset = merged_offset;
                    range.size = merged_end - merged_offset;
                    range.stageFlags |= stage_flags;
                    return true;
                }
            }

            if (end > 256) {
                return false;
            }
            push_constant_ranges.push_back({stage_flags, block_offset, size});
            return true;
        };

        if (!add_push_constant_range(VK_SHADER_STAGE_VERTEX_BIT, 0, vertex_push_constant.total_size) || !add_push_constant_range(VK_SHADER_STAGE_FRAGMENT_BIT, 0, fragment_push_constant.total_size)) {
            return fail("push constant range too big");
        }

        // ---- 5. Descriptor set layouts: parse vertex/fragment separately, then merge by set/binding ----
        auto vertex_descriptor_layout_expected = pipeline::parse_descriptor_set_layouts(vertex_shader_code, VK_SHADER_STAGE_VERTEX_BIT);
        if (!vertex_descriptor_layout_expected) {
            return fail(vertex_descriptor_layout_expected.error());
        }
        const auto vertex_descriptor_layout = std::move(vertex_descriptor_layout_expected).value();

        auto fragment_descriptor_layout_expected = pipeline::parse_descriptor_set_layouts(fragment_shader_code, VK_SHADER_STAGE_FRAGMENT_BIT);
        if (!fragment_descriptor_layout_expected) {
            return fail(fragment_descriptor_layout_expected.error());
        }
        const auto fragment_descriptor_layout = std::move(fragment_descriptor_layout_expected).value();

        const auto merged_layouts = merge_descriptor_layouts(vertex_descriptor_layout, fragment_descriptor_layout);

        resource_guard guard;
        guard.device = device;

        for (const auto& bindings_map : merged_layouts | std::views::values) {
            std::vector<VkDescriptorSetLayoutBinding> bindings;
            bindings.reserve(bindings_map.size());
            for (const auto& binding : bindings_map | std::views::values) {
                bindings.push_back(binding);
            }

            VkDescriptorSetLayoutCreateInfo layout_create_info = {};
            layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout_create_info.bindingCount = static_cast<uint32_t>(bindings.size());
            layout_create_info.pBindings = bindings.data();

            VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
            if (vkCreateDescriptorSetLayout(device, &layout_create_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
                return fail("failed to create descriptor set layout");
            }
            guard.add_set_layout(descriptor_set_layout);
        }

        // ---- 6. pipeline layout ----
        VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
        pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_create_info.pPushConstantRanges = push_constant_ranges.data();
        pipeline_layout_create_info.pushConstantRangeCount = static_cast<uint32_t>(push_constant_ranges.size());
        pipeline_layout_create_info.pSetLayouts = guard.set_layouts.data();
        pipeline_layout_create_info.setLayoutCount = static_cast<uint32_t>(guard.set_layouts.size());

        if (vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr, &guard.pipeline_layout) != VK_SUCCESS) {
            return fail("failed to create pipeline layout");
        }

        // ---- 7. graphics pipeline ----
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
        pipeline_create_info.layout = guard.pipeline_layout;
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
        vk_pipeline result(guard.pipeline, guard.pipeline_layout, guard.set_layouts, device);
        guard.release();
        return result;
    }
} // namespace vulkan
