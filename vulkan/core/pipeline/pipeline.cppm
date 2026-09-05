module;

#include <vulkan/vulkan.h>

export module vulkan.core.pipeline;
export import std;
export import vulkan.core.handles;

/**
 * @file pipeline.cppm
 * @defgroup vulkan_pipeline Vulkan Pipeline
 * @brief graphics pipeline creation from raw SPIR-V binary
 * @note
 *      - make_pipeline() builds a full pipeline from raw SPIR-V binary
 *      - the vertex input layout is parsed from the SPIR-V itself, no user-defined structure needed
 *      - the descriptor set layout and push constant block are NOT parsed: all pipelines share
 *        the agreed flat scene layout owned by core (see core::init_scene_layouts), so no
 *        per-pipeline layout objects exist and one descriptor set works with every pipeline
 *      - returns std::expected<vk_pipeline, std::string_view>, errors carry a message
 */
namespace vulkan {
    /**
     * @ingroup vulkan_pipeline
     * @brief create a graphics pipeline directly from raw SPIR-V binary
     * @param device the logical device
     * @param pipeline_layout the shared scene pipeline layout (owned by core, not the pipeline)
     * @param render_pass the render pass the pipeline renders into; pass VK_NULL_HANDLE to use
     *        dynamic rendering instead (the attachments are then described by color/depth formats)
     * @param color_format swapchain color attachment format (dynamic rendering only)
     * @param depth_format depth attachment format (dynamic rendering only)
     * @param vertex_shader_code raw SPIR-V binary of the vertex shader
     * @param fragment_shader_code raw SPIR-V binary of the fragment shader
     * @param msaa_level MSAA sample count used by the render pass
     * @param depth_test_enabled enable depth test + depth write (false e.g. for the skybox pass)
     * @param has_color_attachment whether the pipeline renders color (false for depth-only
     *        passes like the shadow map: no color attachment, no color blending)
     * @return vk_pipeline on success, error message on failure
     */
    export std::expected<vk_pipeline, std::string_view> make_pipeline(
        VkDevice device,
        VkPipelineLayout pipeline_layout,
        VkRenderPass render_pass,
        VkFormat color_format,
        VkFormat depth_format,
        std::span<unsigned char const> vertex_shader_code,
        std::span<unsigned char const> fragment_shader_code,
        VkSampleCountFlagBits msaa_level,
        bool depth_test_enabled = true,
        bool has_color_attachment = true,
        // fixed-function rasterization depth bias (only meaningful for depth-writing passes
        // such as the shadow map): slope-scaled bias removes shadow acne on angled surfaces
        float depth_bias_constant_factor = 0.0f,
        float depth_bias_slope_factor = 0.0f,
        float depth_bias_clamp = 0.0f);
} // namespace vulkan
