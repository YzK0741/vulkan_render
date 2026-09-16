// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/ssgi_spec.cppm
 * @brief The sixth real pass, and the GI chain's second stage: the glossy lobe, which traces one reflection ray
 *        per pixel and ADD its result to the raw diffuse trace.
 * @defgroup vulkan_pass_ssgi_spec SSGI Specular Pass
 *
 * WHY IT IS A PASS OF ITS OWN AND NOT PART OF THE TRACER, even though it binds the same two shared sets and
 * writes the same image: it is a different dispatch with a different push block, a different reach knob and its
 * own two outputs, and the frame position between them is what the chain's correctness rests on -
 *
 *  * AFTER the tracer, because it READS the raw trace it corrects (a read-after-write between two consecutive
 *    dispatches, which no command buffer gives for free: the compute-storage barrier is the first thing this
 *    pass records);
 *  * BEFORE the denoiser, because the temporal resolve consumes the SUM. That is why the tracer skips its
 *    hand-off barrier when this pass runs (its `specular_next`): the transition to SHADER_READ has to come
 *    after the LAST writer of the image, and this pass is that writer.
 *
 * WHAT IT OWNS: its pipeline layout and its compute pipeline, the dispatch, the ordering barrier, the hand-off
 * barrier, and the per-image state the first-use transition of its two outputs needs. It binds no set of its
 * own and binds no descriptor at all - the same shape as the tracer, reached through the same
 * `pass_io::barrier_images` channel.
 *
 * WHAT IT DELIBERATELY DOES NOT OWN: the instance table's address (the renderer's, and its absence is what makes
 * the pass skip entirely), the frame counter that seeds its ray sequence, its reach and its ray count - all
 * values in the push block the host composes.
 */

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <glm/glm.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>

export module vulkan.pass.ssgi_spec;

import vulkan.pass;
import vulkan.render_resource;
import vulkan.core.handles; // vk_pipeline: the RAII owner of the compute pipeline this pass builds

export namespace vulkan::pass {

    /// @brief what the renderer hands the lobe: nothing but the push block and the images it declared
    /// @note the struct exists so the boundary has a name, not because the pass needs a value from the frame:
    ///       every decision this pass makes is either in `resolved_io` or its own per-image state
    struct ssgi_spec_frame {
        /// how many images the target generation holds, which is how long the per-image first-use state is
        uint32_t image_count = 0;
    };

    /**
     * @brief the glossy lobe: one traced reflection ray per pixel, added to the raw diffuse trace
     *
     * It runs only where the frame can SHADE a hit (the instance table exists) and only when the user asked for
     * the lobe; either being absent means the lighting stage's own specular ambient stands, which is what the
     * frame looked like before this pass existed.
     */
    class ssgi_spec_pass final : public frame_pass {
    public:
        /// @brief the push block, which is also `ssgi_spec.comp`'s
        struct push_constants {
            glm::mat4 inv_view_proj = glm::mat4(1.0f);
            glm::vec4 params = glm::vec4(0.0f);
            /// z/w = the instance table's device address in two halves; 0 means "no table", and the pass skips
            glm::vec4 table = glm::vec4(0.0f);
        };

        ssgi_spec_pass() = default;
        ~ssgi_spec_pass() override;

        [[nodiscard]] render_resource::pass_io const& io() const noexcept override;
        [[nodiscard]] vulkan::pass::behaviour const& behaviour() const noexcept override;
        [[nodiscard]] std::string_view feature() const noexcept override;
        void create(pass_context const& context) override;
        void on_swapchain_recreated(pass_host const& host) override;
        void record(resolved_io const& io) override;

        /// @brief whether the pass built what it records with (the renderer gates the lobe's feature on this)
        [[nodiscard]] bool pipeline_ready() const noexcept;
        /// @brief the pipeline the runner binds before this pass records
        [[nodiscard]] VkPipeline pipeline() const noexcept override;
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept override;

        /**
         * @brief forget that this generation's outputs were ever transitioned
         *
         * The off -> on edge of the GI chain, where the renderer cannot know whether the lobe's two images hold
         * a legal layout: the same reason the TAA resolve is asked to drop its history rather than deciding for
         * itself (see `taa_pass::reset_history`).
         */
        void reset_first_use() noexcept;

        void set_frame(ssgi_spec_frame const& frame) noexcept;

    private:
        static constexpr std::string_view shader_name = "ssgi_spec.comp.spv";
        static constexpr uint32_t group_size = 8; // `ssgi_spec.comp`'s local_size_x/y
        /// the declared barrier images, by the position the declaration gives them
        static constexpr uint32_t barrier_gi_trace = 0;
        static constexpr uint32_t barrier_gi_spec_trace = 1;
        static constexpr uint32_t barrier_gi_spec_reproject = 2;
        static_assert(barrier_gi_spec_reproject + 1 == render_resource::ssgi_spec_barriers.size(),
                      "the lobe's barrier slots must match the declaration it indexes");

        static constexpr std::array<std::string_view, 1> pipeline_names = {"ssgi_spec"};
        inline static constexpr vulkan::pass::behaviour behaviour_ = {
            .kind = behaviour_kind::compute,
            .group_size_x = group_size,
            .group_size_y = group_size,
            .group_size_z = 1,
            .extent = extent_rule::half, // the GI chain's resolution
            .extent_of = resource_id::none,
            .pipelines = pipeline_names,
            .resync_viewport = false,
        };
        void release_owned() noexcept;

        VkDevice device_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pipeline_ = std::nullopt;
        /// one flag per swapchain image: whether that image's two lobe outputs have had their first-use
        /// transition for this target generation
        std::vector<bool> seen_ = {};
        ssgi_spec_frame frame_ = {};
    };

    static_assert(sizeof(ssgi_spec_pass::push_constants) == render_resource::ssgi_spec_io.push->size,
                  "the lobe's declared push block must be the size of the struct the renderer composes");

} // namespace vulkan::pass
