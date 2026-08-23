//
// Created by 小叶 on 2026/7/30.
//

module;

#include <vulkan/vulkan.h>

export module vulkan.core.pipeline;
export import std;
export import vulkan.core.handles;

/**
 * @file pipeline.cppm
 * @defgroup vulkan_pipeline Vulkan Pipeline
 * @brief graphics pipeline creation and shader-related data types
 * @note
 *      - make_pipeline() builds a full pipeline from raw SPIR-V binary
 *      - returns std::expected<vk_pipeline, std::string_view>, errors carry a message
 */
namespace vulkan {

    /**
     * @ingroup vulkan_pipeline
     * @brief shader variable data types
     */
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

    /**
     * @ingroup vulkan_pipeline
     * @brief shader variable storage class
     */
    export enum class shader_storage {
        uniform,
        in,
        out,
        inout,
        buffer,
    };

    /**
     * @ingroup vulkan_pipeline
     * @brief descriptor of a single shader variable
     */
    export struct shader_variable_descriptor {
        shader_data_type type;
        shader_storage storage;
        int array_size;
        int location;
        int binding;
    };

    /**
     * @ingroup vulkan_pipeline
     * @brief pipeline type: graphics or compute
     */
    export enum class pipeline_type {
        graphics,
        compute,
    };

    /**
     * @ingroup vulkan_pipeline
     * @brief aliases for input/output shader variable descriptors
     */
    export using shader_input_descriptor = shader_variable_descriptor;
    export using shader_output_descriptor = shader_variable_descriptor;

    /**
     * @ingroup vulkan_pipeline
     * @brief shader information for pipeline creation
     */
    export struct shader_info {
        std::vector<shader_data_type> in;
        std::vector<shader_data_type> out;
        std::vector<std::vector<shader_data_type>> uniforms;
        std::vector<shader_data_type> push_constant;
        vk_shader_module shader_module;
    };

    /**
     * @ingroup vulkan_pipeline
     * @brief create a pipeline from parsed shader_info (internal use)
     */
    std::optional<vk_pipeline> make_pipeline(
        VkDevice& device,
        VkRenderPass render_pass,
        shader_info const &vertex_shader,
        shader_info const &fragment_shader, VkSampleCountFlagBits msaa_level);

    /**
     * @ingroup vulkan_pipeline
     * @brief create a graphics pipeline directly from raw SPIR-V binary
     * @param device the logical device
     * @param render_pass the render pass the pipeline renders into
     * @param vertex_shader_code raw SPIR-V binary of the vertex shader
     * @param fragment_shader_code raw SPIR-V binary of the fragment shader
     * @param msaa_level MSAA sample count used by the render pass
     * @return vk_pipeline on success, error message on failure
     */
    export std::expected<vk_pipeline, std::string_view> make_pipeline(
        VkDevice &device,
        VkRenderPass render_pass,
        // NOLINT(*-misplaced-const)
        std::vector<unsigned char> vertex_shader_code,
        std::vector<unsigned char> fragment_shader_code,
        VkSampleCountFlagBits msaa_level
    );

}