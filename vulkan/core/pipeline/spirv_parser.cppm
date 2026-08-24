module;

#include <vulkan/vulkan_core.h>

export module vulkan.pipeline.spirv_parser;
export import std;

/**
 * @file spirv_parser.cppm
 * @defgroup vulkan_spirv_parser SPIR-V Reflection Parser
 * @brief parse shader stage interfaces, descriptor set layouts and push constant layouts from SPIR-V binary
 * @note
 *      - built on SPIRV-Reflect
 *      - all functions return std::expected, errors are reported as std::string_view
 */
namespace vulkan::pipeline {
    /**
     * @ingroup vulkan_spirv_parser
     * @brief byte size of the given VkFormat
     * @param format the format to query
     * @return byte size of a single element of the format
     */
    export uint32_t format_size(VkFormat format);

    // 用于存放单个描述符集布局的完整信息
    /**
     * @ingroup vulkan_spirv_parser
     * @brief full information of a single descriptor set layout
     */
    export struct descriptor_set_layout_data {
        uint32_t set_number; // 描述符集索引
        VkDescriptorSetLayoutCreateInfo create_info; // 用于创建 VkDescriptorSetLayout 的信息
        std::vector<VkDescriptorSetLayoutBinding> bindings; // 具体的绑定信息
    };

    /**
     * @ingroup vulkan_spirv_parser
     * @brief a single input/output interface variable of a shader stage
     */
    export struct interface_variable_info {
        uint32_t location;
        uint32_t component;       // 组件索引，默认为0
        VkFormat format;
        std::string name;         // 变量名，用于调试和错误提示
        bool is_builtin;          // 是否为内置变量
    };


    /**
     * @ingroup vulkan_spirv_parser
     * @brief shader stage interface: stage plus its input/output variables
     */
    export struct shader_stage_interface {
        VkShaderStageFlagBits stage;
        std::vector<interface_variable_info> inputs;
        std::vector<interface_variable_info> outputs;
    };

    /**
     * @ingroup vulkan_spirv_parser
     * @brief parse the input/output interface of a shader stage from SPIR-V binary
     * @param spirv_code the SPIR-V binary code
     * @param stage the shader stage to parse
     * @return shader_stage_interface on success, error message on failure
     */
    export std::expected<shader_stage_interface, std::string_view> parse_shader_stage_interface(
        std::span<const unsigned char> spirv_code,
        VkShaderStageFlagBits stage
    );

    bool validate_interface_match(
    const shader_stage_interface& producer,
    const shader_stage_interface& consumer
    );

    /**
     * @ingroup vulkan_spirv_parser
     * @brief parse all descriptor set layouts from SPIR-V binary
     * @param spirv_code the SPIR-V binary code
     * @param shader_stage the shader stage to parse
     * @return vector of descriptor set layout data on success, error message on failure
     */
    export std::expected<std::vector<descriptor_set_layout_data>, std::string_view>
    parse_descriptor_set_layouts(
        std::span<const unsigned char> spirv_code,
        VkShaderStageFlagBits shader_stage
    );

    /**
     * @ingroup vulkan_spirv_parser
     * @brief a single push constant range
     */
    export struct push_constant_info {
        uint32_t offset;
        uint32_t size;
        VkShaderStageFlags stage_flags;
        std::string name;
    };

    // 存储整个着色器的 Push Constant 布局
    /**
     * @ingroup vulkan_spirv_parser
     * @brief push constant layout of the whole shader
     */
    export struct push_constant_layout {
        std::vector<push_constant_info> constants;
        uint32_t total_size;  // 整个 Push Constant 块的总大小
    };

    /**
     * @ingroup vulkan_spirv_parser
     * @brief parse the push constant layout from SPIR-V binary
     * @param spirv_code the SPIR-V binary code
     * @return push_constant_layout on success, error message on failure
     */
    export std::expected<vulkan::pipeline::push_constant_layout, std::string_view> parse_push_constant_layout(
        std::span<const unsigned char> spirv_code
    );
}