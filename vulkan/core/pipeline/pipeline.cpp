//
// Created by 小叶 on 2026/7/30.
//
module;

#include <expected>
#include <optional>
#include <span>
#include <numeric>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <shaderc/shaderc.hpp>

module vulkan.core.pipeline;
import vulkan.pipeline.spirv_parser;


namespace {

    constexpr uint32_t to_multiple_of_4(const uint32_t value) noexcept {
        return ((value + 3) / 4) * 4;
    }

    using vulkan::shader_data_type;
    constexpr uint32_t get_size(const shader_data_type type) {
        switch (type) {
            case shader_data_type::float_t: return sizeof(glm::float32_t);
            case shader_data_type::int_t: return sizeof(glm::int32_t);
            case shader_data_type::uint_t: return sizeof(glm::uint32_t);
            case shader_data_type::bool_t: return sizeof(glm::float32_t);
            case shader_data_type::vec2: return sizeof(glm::vec2);
            case shader_data_type::vec3: return sizeof(glm::vec3);
            case shader_data_type::vec4: return sizeof(glm::vec4);
            case shader_data_type::ivec2: return sizeof(glm::ivec2);
            case shader_data_type::ivec3: return sizeof(glm::ivec3);
            case shader_data_type::ivec4: return sizeof(glm::ivec4);
            case shader_data_type::bvec2: return sizeof(glm::bvec2);
            case shader_data_type::bvec3: return sizeof(glm::bvec3);
            case shader_data_type::bvec4: return sizeof(glm::bvec4);
            case shader_data_type::mat2: return sizeof(glm::mat2);
            case shader_data_type::mat3: return sizeof(glm::mat3);
            case shader_data_type::mat4: return sizeof(glm::mat4);
            case shader_data_type::mat2x2: return sizeof(glm::mat2x2);
            case shader_data_type::mat2x3: return sizeof(glm::mat2x3);
            case shader_data_type::mat3x2: return sizeof(glm::mat3x2);
            case shader_data_type::mat2x4: return sizeof(glm::mat2x4);
            case shader_data_type::mat3x4: return sizeof(glm::mat3x4);
            case shader_data_type::mat4x2: return sizeof(glm::mat4x2);
            case shader_data_type::mat4x3: return sizeof(glm::mat4x3);
            default: return 1;
        }
    }

    constexpr VkFormat to_vertex_format(const shader_data_type type) {
        switch (type) {
            case shader_data_type::float_t: return VK_FORMAT_R32_SFLOAT;
            case shader_data_type::int_t: return VK_FORMAT_R32_SINT;
            case shader_data_type::uint_t: return VK_FORMAT_R32_UINT;
            case shader_data_type::bool_t: return VK_FORMAT_R32_SFLOAT;
            case shader_data_type::vec2: return VK_FORMAT_R32G32_SFLOAT;
            case shader_data_type::vec3: return VK_FORMAT_R32G32B32_SFLOAT;
            case shader_data_type::vec4: return VK_FORMAT_R32G32B32A32_SFLOAT;
            case shader_data_type::ivec2: return VK_FORMAT_R32G32_SINT;
            case shader_data_type::ivec3: return VK_FORMAT_R32G32B32_SINT;
            case shader_data_type::ivec4: return VK_FORMAT_R32G32B32A32_SINT;
            case shader_data_type::uvec2: return VK_FORMAT_R32G32_UINT;
            case shader_data_type::uvec3: return VK_FORMAT_R32G32B32_UINT;
            case shader_data_type::uvec4: return VK_FORMAT_R32G32B32A32_UINT;
            default: return VK_FORMAT_UNDEFINED;
        }

    }
}

namespace vulkan {
    std::optional<VkShaderModule> make_shader_module(std::span<unsigned char> shader, const VkDevice device) { // NOLINT(*-misplaced-const)
        if (shader.empty()) {
            return std::nullopt;
        }
        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = shader.size();
        create_info.pCode = reinterpret_cast<uint32_t*>(shader.data());
        VkShaderModule shader_module = {};
        if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
            return std::nullopt;
        }
        return shader_module;
    }

    std::optional<vk_pipeline> make_pipeline(
        VkDevice &device, const VkRenderPass render_pass, // NOLINT(*-misplaced-const)
        shader_info const &vertex_shader,
        shader_info const &fragment_shader, VkSampleCountFlagBits msaa_level) {

        VkVertexInputBindingDescription vertex_input_binding = {};
        vertex_input_binding.binding = 0;
        vertex_input_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        vertex_input_binding.stride = std::accumulate(vertex_shader.in.begin(), vertex_shader.in.end(), 0,
            [](const uint32_t acc, const shader_data_type type) {
            return acc + get_size(type);
        });

        std::vector<VkVertexInputAttributeDescription> attribute_descriptions = {};
        attribute_descriptions.resize(vertex_shader.in.size());
        uint32_t location = 0;
        uint32_t offset = 0;

        for (auto const& attribute : vertex_shader.in) {
            attribute_descriptions[location].location = location;
            attribute_descriptions[location].offset = offset;
            attribute_descriptions[location].format = to_vertex_format(attribute);
            attribute_descriptions[location].binding = 0;
            location++;
            offset += get_size(attribute);
        }

        VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {};
        vertex_input_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_state_create_info.vertexBindingDescriptionCount = 1;
        vertex_input_state_create_info.pVertexBindingDescriptions = &vertex_input_binding;
        vertex_input_state_create_info.vertexAttributeDescriptionCount = attribute_descriptions.size();
        vertex_input_state_create_info.pVertexAttributeDescriptions = attribute_descriptions.data();

        VkPipelineShaderStageCreateInfo vertex_stage = {};
        vertex_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertex_stage.module = *vertex_shader.shader_module;
        vertex_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertex_stage.pName = "main";
        vertex_stage.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo fragment_stage = {};
        fragment_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragment_stage.module = *fragment_shader.shader_module;
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

        std::vector<VkPushConstantRange> push_constant_range = {};
        if (!vertex_shader.push_constant.empty()) {
            push_constant_range.emplace_back();
            push_constant_range.back().stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            push_constant_range.back().offset = 0;
            push_constant_range.back().size = to_multiple_of_4(std::accumulate(vertex_shader.push_constant.begin(), vertex_shader.push_constant.end(), 0u,
                [](const uint32_t acc, const shader_data_type type) {
                    return acc + get_size(type);
                }
            ));
        }

        if (!fragment_shader.push_constant.empty()) {
            push_constant_range.emplace_back();
            push_constant_range.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            push_constant_range.back().offset = push_constant_range.size() == 2? push_constant_range[0].size : 0;
            push_constant_range.back().size = to_multiple_of_4(std::accumulate(fragment_shader.push_constant.begin(), fragment_shader.push_constant.end(), 0u,
                [](const uint32_t acc, const shader_data_type type) {
                    return acc + get_size(type);
                }
            ));
        }

        if (!push_constant_range.empty() && push_constant_range.back().offset + push_constant_range.back().size > 256) {
            return std::nullopt;
        }

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.resize(2);
        // 这里要求两个shader的uniform是相同的,且为2，1为ubo，2为纹理，否则视作违反调用规定
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[0].pImmutableSamplers = nullptr;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layout_create_info = {};
        layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_create_info.bindingCount = 2;
        layout_create_info.pBindings = bindings.data();

        VkDescriptorSetLayout descriptor_set_layout = {};
        if (vkCreateDescriptorSetLayout(device, &layout_create_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
            return std::nullopt;
        }


        VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
        pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_create_info.pPushConstantRanges = push_constant_range.data();
        pipeline_layout_create_info.pushConstantRangeCount = push_constant_range.size();
        pipeline_layout_create_info.pSetLayouts = &descriptor_set_layout;
        pipeline_layout_create_info.setLayoutCount = 1;

        VkPipelineLayout pipeline_layout;

        if (vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
            vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);

            return std::nullopt;
        }

        VkGraphicsPipelineCreateInfo pipeline_create_info = {};
        pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
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

        VkPipeline pipeline;
        if (vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_create_info, nullptr, &pipeline) != VK_SUCCESS) {
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
            return std::nullopt;
        }

        return vk_pipeline(pipeline, pipeline_layout, descriptor_set_layout, device);
    }

    std::expected<vk_pipeline, std::string_view> make_pipeline(
        [[maybe_unused]] const VkDevice device, // NOLINT(*-misplaced-const)
        [[maybe_unused]] const VkRenderPass renderpass, // NOLINT(*-misplaced-const)
        [[maybe_unused]] std::vector<unsigned char> const &vertex_shader_code,
        [[maybe_unused]] std::vector<unsigned char> const &fragment_shader_code,
        [[maybe_unused]] VkSampleCountFlagBits msaa_level
    ) {

        using fail = std::unexpected<std::string_view>;

        auto vertex_interface_expected = pipeline::parse_shader_stage_interface(vertex_shader_code, VK_SHADER_STAGE_VERTEX_BIT);

        if (!vertex_interface_expected) {
            return fail(vertex_interface_expected.error());
        }

        auto vertex_interface = std::move(vertex_interface_expected).value();

        VkVertexInputBindingDescription vertex_input_binding = {};
        vertex_input_binding.binding = 0;
        vertex_input_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        vertex_input_binding.stride = std::accumulate(vertex_interface.inputs.begin(), vertex_interface.inputs.end(), 0,
            [](const uint32_t acc, decltype(*vertex_interface.inputs.begin()) type) {
            return acc + pipeline::format_size(type.format);
        });

        std::vector<VkVertexInputAttributeDescription> attribute_descriptions = {};
        attribute_descriptions.resize(vertex_interface.inputs.size());
        uint32_t location = 0;
        uint32_t offset = 0;

        for (uint64_t index = 0; index < vertex_interface.inputs.size(); index++) {
            auto& attribute = attribute_descriptions[index];
            const auto& variable = vertex_interface.inputs[index];
            attribute.format = variable.format;
            attribute.location = location;
            attribute.offset = offset;
            location++;
            offset += pipeline::format_size(variable.format);
        }


        return fail("function not completed");
    }
}

