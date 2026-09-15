// module version: 0.2.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/ssgi_temporal.cppm
 * @brief The seventh real pass: the GI chain's temporal resolve for the DIFFUSE bounce.
 * @defgroup vulkan_pass_ssgi_temporal SSGI Temporal Resolve Pass
 *
 * WHAT IT OWNS: the frame's RECORDING - the two barrier batches, the dispatch, the push lanes that describe its
 * own state, the history copy and the hand-backs - which is the part that was buried in
 * `runtime::record_ssgi_resolve_pass`; and, since this step, the SET LAYOUT its declaration generates, the
 * PIPELINE LAYOUT and the COMPUTE PIPELINE built from them and from its own shader. The renderer used to build
 * all three (`runtime::make_ssgi_temporal_pipeline`), which is why they left: a handle only this pass names is
 * this pass's to build and to release.
 *
 * WHAT IT STILL DOES NOT OWN, and the reason is a framework gap rather than an oversight: its per-image
 * DESCRIPTOR FAMILY. Two signals are resolved through this one layout and pipeline - the diffuse bounce (this
 * pass) and the reflection (the renderer's `record_ssgi_resolve_pass`, mode 1) - and each needs its own list of
 * images in the same seven slots. So the renderer goes on building BOTH families, taking this pass's set layout
 * through `set_layout()`, which is the honest half of the split: the layout is the pass's, the family that a
 * second signal also uses is not. `resolved_io::own_per_image` (added for exactly this) is what the rest of the
 * move needs - see docs/pass_chain_plan.md.
 *
 * WHY THE SPLIT IS WORTH TAKING ANYWAY: the recording is where the ORDER lives - the resolve must run after the
 * tracer and the lobe (both write the raw trace it reads) and before the spatial filter (which consumes its
 * output) - and that order is now a property of a pass rather than of a 170-line function in the frame loop.
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
#include <vulkan/vulkan.h>

export module vulkan.pass.ssgi_temporal;

import vulkan.pass;
import vulkan.render_resource;
import vulkan.bindings;     // make_set_layout: a declaration generates this pass's own set layout
import vulkan.core.handles; // vk_pipeline: the RAII owner of the compute pipeline this pass builds

export namespace vulkan::pass {

    /// @brief what the renderer hands the diffuse temporal resolve
    struct ssgi_temporal_frame {
        /**
         * Whether the accumulation may be blended with, AS READ BEFORE this dispatch.
         *
         * The renderer snapshots it, so the REFLECTION's resolve - which runs after this one in the same frame
         * - blends exactly as this one did. A flag read after this dispatch would tell the reflection its history
         * exists on the very frame that created it, which is the one frame it must not blend with.
         */
        bool history_valid = false;
        /**
         * The renderer's two shared per-image transitions (the G-buffer depth and the motion-vector target),
         * written for this dispatch only: their "was it written this frame" flags belong to the passes that WROTE
         * those images, so the pass cannot own them - the same shape as the scene pass's `make_environment`.
         */
        void (*ensure_inputs)(void* owner, VkCommandBuffer command_buffer, uint32_t image_index) = nullptr;
        void* owner = nullptr;
    };

    /**
     * @brief the diffuse temporal resolve: blend this frame's trace with the accumulation of earlier frames
     *
     * It is the middle of the GI chain: after the tracer and the lobe (which write the image it reads) and before
     * the spatial filter, whose output the composite samples.
     */
    class ssgi_temporal_pass final : public frame_pass {
    public:
        /// @brief the push block, which is also `ssgi_temporal.comp`'s
        struct push_constants {
            float history_valid = 0.0f;
            float blend_static = 0.9f;
            float blend_min = 0.6f;
            float depth_scale = 0.0f;
            float depth_offset = 0.0f;
            /// 1.0 = the reflection, 0.0 = the diffuse bounce (the shader's `glossy`); this pass pushes 0
            float mode = 0.0f;
            float unused1 = 0.0f;
            float unused2 = 0.0f;
            glm::vec4 gi_size = glm::vec4(0.0f); // xy = GI extent, zw = full-res extent
        };

        ssgi_temporal_pass() = default;
        ~ssgi_temporal_pass() override;

        [[nodiscard]] render_resource::pass_io const& io() const noexcept override;
        [[nodiscard]] vulkan::pass::behaviour const& behaviour() const noexcept override;
        [[nodiscard]] std::string_view feature() const noexcept override;
        void create(pass_context const& context) override;
        void on_swapchain_recreated(pass_host const& host) override;
        void record(resolved_io const& io) override;

        /// @brief whether the accumulation was written this frame (what the spatial filter waits for)
        [[nodiscard]] bool resolved() const noexcept;

        /// @brief whether the pass built what it records with (the renderer gates the GI chain on this)
        [[nodiscard]] bool pipeline_ready() const noexcept;
        /// @brief the pipeline the runner binds before this pass records
        [[nodiscard]] VkPipeline pipeline() const noexcept;
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept;
        /**
         * @brief the set layout this pass's declaration generates
         *
         * Handed out because a SECOND signal (the reflection) is resolved through the same layout by the
         * renderer, which builds that family itself - see the header for why that family is not this pass's.
         */
        [[nodiscard]] VkDescriptorSetLayout set_layout() const noexcept;

        void set_frame(ssgi_temporal_frame const& frame) noexcept;

    private:
        static constexpr std::string_view shader_name = "ssgi_temporal.comp.spv";
        static constexpr uint32_t group_size = 8; // `ssgi_temporal.comp`'s local_size_x/y
        /// the declared barrier images, by the position the declaration gives them
        static constexpr uint32_t barrier_resolve = 0;
        static constexpr uint32_t barrier_history = 1;
        static_assert(barrier_history + 1 == render_resource::ssgi_temporal_barriers.size(),
                      "the resolve's barrier slots must match the declaration it indexes");

        static constexpr std::array<std::string_view, 1> pipeline_names = {"ssgi_temporal"};
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

        bool resolved_ = false;
        VkDevice device_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pipeline_ = std::nullopt;
        ssgi_temporal_frame frame_ = {};
    };

    static_assert(sizeof(ssgi_temporal_pass::push_constants) == render_resource::ssgi_temporal_io.push->size,
                  "the resolve's declared push block must be the size of the struct the renderer composes");

} // namespace vulkan::pass
