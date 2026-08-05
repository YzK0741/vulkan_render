//
// Created by 小叶 on 2026/7/30.
//

module;

#include <optional>
#include <vector>
#include <expected>
#include <string_view>
#include <vulkan/vulkan.h>

export module vulkan.core.pipeline;

export import vulkan.core.handles;

namespace vulkan {

    export enum class shader_data_type {
        float_t, int_t, uint_t, bool_t,
        vec2, vec3, vec4,
        ivec2, ivec3, ivec4,
        uvec2, uvec3, uvec4,
        bvec2, bvec3, bvec4,
        mat2, mat3, mat4,
        mat2x2 ,mat2x3, mat3x2, mat2x4, mat4x2, mat3x4, mat4x3,
        sampler2D, samplerCube, sampler3D, sampler2DArray, sampler2DShadow,
        image2D, image3D, imageCube, image2DArray,
    };

    export enum class shader_storage {
        uniform,
        in,
        out,
        inout,
        buffer,
    };

    export struct shader_variable_descriptor {
        shader_data_type type;
        shader_storage storage;
        int array_size;
        int location;
        int binding;
    };

    export enum class pipeline_type {
        graphics,
        compute,
    };

    export using shader_input_descriptor = shader_variable_descriptor;
    export using shader_output_descriptor = shader_variable_descriptor;

    export struct shader_info {
        std::vector<shader_data_type> in;
        std::vector<shader_data_type> out;
        std::vector<std::vector<shader_data_type>> uniforms;
        std::vector<shader_data_type> push_constant;
        vk_shader_module shader_module;
    };

    std::optional<vk_pipeline> make_pipeline(
        VkDevice& device,
        VkRenderPass render_pass,
        shader_info const &vertex_shader,
        shader_info const &fragment_shader, VkSampleCountFlagBits msaa_level);

    export std::expected<vk_pipeline, std::string_view> make_pipeline(VkDevice device, VkRenderPass renderpass,
                                                               std::vector<unsigned char> const &vertex_shader_code,
                                                               std::vector<unsigned char> const &fragment_shader_code,
                                                               VkSampleCountFlagBits msaa_level);

    export std::expected<vk_pipeline, std::string_view> make_pipeline(
        VkDevice &device,
        VkRenderPass render_pass,
        // NOLINT(*-misplaced-const)
        std::vector<unsigned char> vertex_shader_code,
        std::vector<unsigned char> fragment_shader_code,
        VkSampleCountFlagBits msaa_level
    );

}