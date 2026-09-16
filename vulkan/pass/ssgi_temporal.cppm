// module version: 0.5.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/ssgi_temporal.cppm
 * @brief The seventh real pass: the GI chain's temporal resolve for the DIFFUSE bounce.
 * @defgroup vulkan_pass_ssgi_temporal SSGI Temporal Resolve Pass
 *
 * WHAT IT OWNS: the frame's RECORDING - the two barrier batches, the dispatch, the push lanes that describe its
 * own state, the history copy and the hand-backs - which is the part that was buried in
 * `runtime::record_ssgi_resolve_pass`; the SET LAYOUT its declaration generates and the PIPELINE LAYOUT and
 * COMPUTE PIPELINE built from it; and the DESCRIPTOR FAMILY that set layout's per-image sets need, written from
 * `resolved_io::own_per_image` (the channel that was added for exactly this pass). The renderer used to build all
 * of them, which is why they left: a handle only this pass names is this pass's to build, to write and to
 * release - and the family is the one that proves the point, because writing image k's set needs image k's views
 * and only the pass may reach for them (`bindings::write_set` generates the writes from the declaration, so the
 * pass decides the views and nothing else).
 *
 * WHAT IT STILL DOES NOT OWN, and the reason is a framework gap rather than an oversight: the REFLECTION's
 * family and the mode-1 recording that uses it. Two signals are resolved through this one layout and pipeline -
 * the diffuse bounce (this pass) and the reflection (the renderer's `record_ssgi_resolve_pass`) - and each needs
 * its own list of images in the same seven slots, which a single declaration cannot describe. The renderer builds
 * that second family on this pass's layout, taken through `set_layout()`; see docs/pass_chain_plan.md.
 *
 * ... AND ITS FRAME'S `record_reflection` IS THE SAME GAP (the reflection is called back from inside this pass's
 * recording, so the two stay contiguous), while the two shared per-image transitions it used to be handed are NOT:
 * those are the FRAME's ordering rules and run in the frame loop, between the chain's two halves - see
 * `runtime::record_main_drawcalls` and docs/pass_chain_plan.md.
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
import vulkan.render_resource.shared; // sampler_set: the handles bindings::write_set binds
import vulkan.bindings;               // make_set_layout / write_set / image_set_family: this pass's own set
import vulkan.core.handles;           // vk_pipeline: the RAII owner of the compute pipeline this pass builds

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
         * The REFLECTION's recording, which is the renderer's because a declaration cannot describe two signals
         * in the same seven slots (see the file's header): the pass calls this at the END of its own recording -
         * after mode 0's hand-backs, before the spatial filter that reads both accumulations - so that the GI
         * chain stays CONTIGUOUS (`vulkan.pass.chain`) instead of being split around a runtime call.
         *
         * The callback is handed the SAME `history_valid` the pass itself used, because the two signals must
         * agree about the frame that created the history: a flag read after mode 0's dispatch would tell the
         * reflection its history exists on the very frame that created it.
         */
        void (*record_reflection)(void* owner, VkCommandBuffer command_buffer, bool history_valid) = nullptr;
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

        /**
         * The two blend weights are CONSTANTS rather than a parameter, and that is the measured answer rather
         * than a shortcut: the GI signal is far noisier than shading aliasing, so it wants a much longer memory,
         * and it must not be tuned by whatever the AA sliders happen to be set to. Nothing has ever set them.
         *
         * PUBLIC because a second recording shares this pipeline: the renderer's mode-1 (the reflection) composes
         * the same block, and both lanes have to be the same number or the two signals are denoised differently.
         */
        static constexpr float blend_static = 0.9f;
        static constexpr float blend_min = 0.6f;

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
        /// @brief the framework's generic form of the same question, so an owner holding only a chain can ask it
        ///        (a pass with nothing of its own to build keeps the interface's 	rue; see rame_pass::ready)
        [[nodiscard]] bool ready() const noexcept override {
            return this->pipeline_ready();
        }
        /// @brief the pipeline the runner binds before this pass records
        [[nodiscard]] VkPipeline pipeline() const noexcept override;
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept override;
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

        /// the two declared bindings this pass has to know by name (see the substitution in record)
        static constexpr std::size_t binding_gbuffer_depth = 3;
        static constexpr std::size_t binding_spec_reproject = 6;
        static_assert(binding_spec_reproject + 1 == render_resource::ssgi_temporal_io.bindings.size(),
                      "the resolve's named bindings must match the declaration it indexes");

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
        render_resource::shared::sampler_set samplers_ = {};
        VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pipeline_ = std::nullopt;
        /// the per-image sets this pass writes itself, one set per swapchain image (see the header)
        bindings::image_set_family family_ = {};
        ssgi_temporal_frame frame_ = {};
    };

    static_assert(sizeof(ssgi_temporal_pass::push_constants) == render_resource::ssgi_temporal_io.push->size,
                  "the resolve's declared push block must be the size of the struct the renderer composes");

} // namespace vulkan::pass
