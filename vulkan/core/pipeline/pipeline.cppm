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
 *      - the vertex input layout, descriptor set layouts and push constant ranges
 *        are all parsed from the SPIR-V itself, no user-defined structure needed
 *      - returns std::expected<vk_pipeline, std::string_view>, errors carry a message
 */
namespace vulkan {
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
        VkDevice& device,
        VkRenderPass render_pass,
        std::span<const unsigned char> vertex_shader_code,
        std::span<const unsigned char> fragment_shader_code,
        VkSampleCountFlagBits msaa_level);
} // namespace vulkan
