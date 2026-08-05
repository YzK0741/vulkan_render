//
// Created by 小叶 on 2026/8/2.
//

module;

#include <expected>
#include <vector>
#include <string_view>
#include <string>
#include <span>
#include <vulkan/vulkan_core.h>

export module vulkan.pipeline.spirv_parser;

namespace vulkan::pipeline {
    export uint32_t format_size(VkFormat format);

    // 用于存放单个描述符集布局的完整信息
    export struct descriptor_set_layout_data {
        uint32_t set_number; // 描述符集索引
        VkDescriptorSetLayoutCreateInfo create_info; // 用于创建 VkDescriptorSetLayout 的信息
        std::vector<VkDescriptorSetLayoutBinding> bindings; // 具体的绑定信息
    };

    export struct interface_variable_info {
        uint32_t location;
        uint32_t component;       // 组件索引，默认为0
        VkFormat format;
        std::string name;         // 变量名，用于调试和错误提示
        bool is_builtin;          // 是否为内置变量
    };


    export struct shader_stage_interface {
        VkShaderStageFlagBits stage;
        std::vector<interface_variable_info> inputs;
        std::vector<interface_variable_info> outputs;
    };

    export std::expected<shader_stage_interface, std::string_view> parse_shader_stage_interface(
        std::span<unsigned char> spirv_code,
        VkShaderStageFlagBits stage
    );

    bool validate_interface_match(
    const shader_stage_interface& producer,
    const shader_stage_interface& consumer
    );

    export std::expected<std::vector<descriptor_set_layout_data>, std::string_view>
    parse_descriptor_set_layouts(
        std::span<unsigned char> spirv_code,
        VkShaderStageFlagBits shader_stage
    );

    export struct push_constant_info {
        uint32_t offset;
        uint32_t size;
        VkShaderStageFlags stage_flags;
        std::string name;
    };

    // 存储整个着色器的 Push Constant 布局
    export struct push_constant_layout {
        std::vector<push_constant_info> constants;
        uint32_t total_size;  // 整个 Push Constant 块的总大小
    };

    export std::expected<vulkan::pipeline::push_constant_layout, std::string_view> parse_push_constant_layout(
        std::span<unsigned char> spirv_code
    );
}