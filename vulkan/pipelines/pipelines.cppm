// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pipelines/pipelines.cppm
 * @brief The engine's per-pass pipelines: one builder per pass, next to the generic builder in
 *        vulkan.core.pipeline (that one knows HOW to build a pipeline, this one knows what each pass's
 *        pipeline looks like - formats, sample counts, blend state, push-constant ranges).
 * @ingroup vulkan_pipelines
 *
 * Extracted from vulkan.runtime, whose implementation had grown past 4900 lines. The builders are
 * stateless: the caller passes the core and the set layouts the pass reuses and gets the created
 * handles back. Ownership stays with the runtime, which keeps the members and the call ORDER - the
 * order matters because the set layout a pass needs is owned by the pass that creates it (fxaa needs
 * the post layout, deferred the G-buffer layout, and both say so in their error messages).
 *
 * The pipeline handles are std::optional because vk_pipeline is an RAII owner with no default
 * constructor, which is also how the runtime holds them.
 */

module;

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

export module vulkan.pipelines;

import vulkan.core;
import vulkan.core.pipeline; // vk_pipeline

namespace vulkan::pipelines {
    /// what build_post() creates: the post set layout, its pipeline layout and the two composites
    export struct post_owned {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> composite; // tonemap + bloom sum, writes the swapchain
        std::optional<vk_pipeline> hdr;       // the same pass writing an HDR target instead (FXAA on)
    };

    /// what build_gbuffer_debug() creates: the G-buffer set layout (owned here), its pipeline layout
    /// and the debug view pipeline
    export struct gbuffer_owned {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> debug;
    };

    /// what build_taa() creates
    export struct taa_owned {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        std::optional<vk_pipeline> resolve;
    };

    export std::expected<post_owned, std::string> build_post(core& vk, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export std::expected<gbuffer_owned, std::string> build_gbuffer_debug(core& vk, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export std::expected<taa_owned, std::string> build_taa(core& vk, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);

    /// the passes that reuse a layout someone else owns, so theirs comes in as a parameter
    export std::expected<vk_pipeline, std::string> build_fxaa(core& vk, VkPipelineLayout post_pipeline_layout, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
    export std::expected<vk_pipeline, std::string> build_deferred(core& vk, VkPipelineLayout pipeline_layout, std::span<unsigned char const> vertex_shader_code, std::span<unsigned char const> fragment_shader_code);
} // namespace vulkan::pipelines