// module version: 0.2.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/post.cppm
 * @brief The twelfth and thirteenth real passes: the POST CHAIN - the composite and the bloom chain's four levels.
 * @defgroup vulkan_pass_post Post Chain Passes
 *
 * WHY THEY ARE ONE MODULE AND FIVE PASSES. They are one module because they are one shader: `post.frag` declares
 * ONE push constant block and selects its stage with a `mode` lane (0 bright-pass prefilter, 1 downsample,
 * 2 composite, 3 FXAA), so the block's SHAPE is shared and belongs next to the passes that push it - not in the
 * renderer, where it lived while the renderer owned the recordings. They are five passes because the framework
 * hands a pass ONE descriptor set per shared owner, and each of the five stages binds a DIFFERENT set of the post
 * family (the prefilter reads the HDR target, downsample L reads level L-1, the composite reads all four levels
 * plus the GI image and the G-buffer depth and normal).
 *
 * WHAT THE COMPOSITE OWNS, and it is the whole post chain's GPU material: the post SET LAYOUT (nine combined
 * image samplers), the pipeline layout that binds it and takes the 52-byte push block, and the TWO pipelines the
 * chain records with - one per colour format the chain renders into, because a pipeline's declared colour format
 * has to match the attachment it renders into (the swapchain's format for the composite when it finishes the
 * frame, R16F for the bloom levels and for the composite when FXAA will finish it instead). `pipelines::build_post`
 * is what creates all four objects in one call, which is why the bloom passes reach the layout through this pass
 * rather than building a second copy of it.
 *
 * WHAT THE BLOOM PASSES OWN: their own recording - the transition of the level they read, the clear-instance over
 * the level they write, the one-lane push and the fullscreen draw - and the deepest level's HAND-BACK (it moves
 * its own output to a sampled layout, because the composite samples all four levels and the last level has no
 * successor to do it for it, the same writer's hand-back the GI chain's passes use). They build no pipeline: the
 * runner binds the chain's R16F variant, which their `behaviour` names and the host resolves (the shape the
 * temporal resolve's first extraction had, and the reason it needed no new framework).
 *
 * WHAT NEITHER OWNS, recorded because it is a decision rather than an omission: the HDR target's transition to a
 * sampled layout. Two of these passes read that image and it must be moved exactly once on frames where the bloom
 * chain does NOT run at all, so its single owner is the frame loop - the same "shared per-image transition" split
 * the deferred lighting stage's input transitions have.
 */

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vulkan/vulkan.h>

export module vulkan.pass.post;

import vulkan.pass;
import vulkan.render_resource;
import vulkan.core.handles; // vk_pipeline: the RAII owner of the pipelines this pass builds

export namespace vulkan::pass {

    /**
     * @brief the post chain's ONE push block, which is also `post.frag`'s
     *
     * Thirteen floats in the order the shader declares them, and every stage pushes ALL of them: the lanes a
     * stage does not read are part of the bytes the renderer pushed before these passes existed, and the
     * non-zero defaults below (the GI upsample's sigma and normal power, its on switch, the FXAA knobs) are that
     * history. `mode` is the one lane a PASS writes - it is the stage's own identity - and the rest are the
     * frame's values.
     */
    struct post_push_constants {
        float exposure = 1.0f;        // linear exposure scale (see set_exposure)
        float bloom_intensity = 0.0f; // bloom blend weight (see set_bloom)
        float bloom_threshold = 0.0f; // bloom bright-pass threshold (see set_bloom)
        float mode = 0.0f;            // 0 = prefilter, 1 = downsample, 2 = composite, 3 = FXAA
        // composite only: 1 = the shader encodes to sRGB itself, 0 = the target is an sRGB attachment and the
        // hardware encodes on write. Filled from the frame's TARGET (never hard-coded: either choice
        // double-encodes an sRGB attachment or under-encodes a UNORM one), and the FXAA pass forces it to 1,
        // because it renders into the R16F LDR image, which must hold gamma-encoded values for FXAA's luma
        // thresholds.
        float encode_gamma = 0.0f;
        // composite only: the weight on the screen-space GI image (1 = GI ran this frame, 0 = the added term is
        // exactly zero, which is what keeps GI-off frames byte-identical). The tracer scales the signal itself;
        // this is the on/off switch folded into the push block rather than a shader branch on a feature flag it
        // cannot see.
        float gi_intensity = 0.0f;
        // composite only: the joint-bilateral UPSAMPLE of the half-resolution GI. A plain bilinear fetch of a
        // half-resolution image mixes in the neighbouring texels across a silhouette, which darkens the geometry
        // side and spills light onto the background side; these four terms are the same edge criterion the
        // spatial filter uses, so a silhouette that survives one pass is not re-blurred by the next.
        float gi_depth_scale = 0.0f;  // projection[2][2]
        float gi_depth_offset = 0.0f; // projection[3][2]
        float gi_depth_sigma = 0.02f;
        float gi_normal_power = 16.0f;
        // 1 = the bilateral gather, 0 = the plain bilinear fetch. The off switch exists to make the upsample
        // measurable (the same reason ssgi_spatial_sigma can be 0); it is not a quality knob.
        float gi_upsample = 1.0f;
        // FXAA lanes (fxaa.frag): the sub-pixel term strength (0 = pure directional blend) and the relative luma
        // contrast below which a pixel counts as flat.
        float fxaa_subpixel = 0.75f;
        float fxaa_edge_threshold = 0.166f;
    };

    static_assert(sizeof(post_push_constants) == render_resource::post_push_bytes,
                  "the post chain's declared push block must be the size of the struct its passes push");

    /**
     * @brief what the renderer hands the composite: the overlay's draw, and nothing else
     *
     * The push block does NOT travel here: it travels through `resolved_io::push`, exactly as every other pass's
     * does - the host composes a VALUE of this module's `post_push_constants` (so the shape is still the passes',
     * which is what the static_assert above is for) and the pass writes only the lane that is its own stage's
     * identity.
     */
    struct composite_frame {
        /**
         * The debug overlay's draw, recorded INSIDE this pass's rendering instance between its draw and
         * `vkCmdEndRendering`.
         *
         * WHY IT IS A CALLBACK AND NOT A PASS OF ITS OWN, which is the decision this step had to make: the
         * overlay has no load op - it composites a UI on top of the image the composite just wrote - so a pass
         * of its own would CLEAR the frame it is supposed to draw over. It has to be inside whichever instance
         * is the frame's LAST writer, and which one that is depends on FXAA: the host sets this callback on the
         * frames FXAA is off and leaves it null when the FXAA pass (vulkan.pass.fxaa, its own step) carries the
         * overlay instead.
         */
        void (*after_draw)(void* owner, VkCommandBuffer command_buffer) = nullptr;
        void* owner = nullptr;
        /**
         * Whether the FXAA pass finishes this frame, i.e. whether the composite must write the LDR image instead
         * of the swapchain (see the target deviation in `render_resource::post_composite_io`).
         *
         * THE HOST'S ANSWER, and it is the SAME predicate that decides `after_draw`'s owner above: the frame's
         * last writer carries the overlay and the other one writes the intermediate. It arrives as data rather
         * than being derived from `after_draw != nullptr`, because a pass inferring one decision from another
         * decision's nullness is exactly the kind of coupling a frame struct exists to prevent.
         */
        bool write_ldr = false;
        /**
         * Whether this frame must NOT add the bloom sum - true while the G-buffer debug view is up.
         *
         * Bloom is a display effect, and a glow smeared over the channel being inspected is the opposite of a debug
         * view (it would also invent colours that are not in the G-buffer at all). The host owns the answer
         * because it is the same "what runs this frame" struct that gates the bloom CHAIN (`feature_active("bloom")`),
         * so the weight this pass adds and the chain's own gate cannot disagree about whether there is a sum.
         */
        bool suppress_bloom = false;
    };

    /**
     * @brief the composite: HDR plus the weighted bloom levels, tonemapped into the frame's display target
     *
     * The frame's LAST writer when FXAA is off (it then carries the overlay through `after_draw`) and the
     * second-to-last when FXAA is on (it writes the R16F LDR image instead, with the R16F pipeline, and the
     * FXAA pass finishes). Which of the two it is, is the host's answer - see `render_resource::post_composite_io`
     * for the recorded target deviation.
     */
    class post_composite_pass final : public frame_pass {
    public:
        /// @brief the NAME the bloom levels declare to record with this pass's R16F variant (see named_pipeline)
        static constexpr std::string_view bloom_pipeline_name = "post_hdr";

        post_composite_pass() = default;
        ~post_composite_pass() override;

        [[nodiscard]] render_resource::pass_io const& io() const noexcept override;
        [[nodiscard]] vulkan::pass::behaviour const& behaviour() const noexcept override;
        [[nodiscard]] std::string_view feature() const noexcept override;
        void create(pass_context const& context) override;
        void on_swapchain_recreated(pass_host const& host) override;
        /**
         * @brief resolve the declaration, then let the FRAME decide the target and the pipeline variant
         *
         * THE ONE POST PASS THAT OVERRIDES `resolve`, and the reason is the deviation its declaration records:
         * `render_target` names one resource, and this pass renders into the SWAPCHAIN when FXAA is off and into
         * the R16F LDR image when it is on (FXAA has to READ what the composite produced, and a pass may not read
         * the image it renders into). The two choices ARE one choice - the target decides the pipeline's format -
         * so they are made together here, from `composite_frame::write_ldr`, which is the frame's answer (the same
         * predicate that decides who draws the overlay).
         *
         * The PUSH BLOCK is composed here too rather than in `record`, because `encode_gamma` is a consequence of
         * the same decision (with FXAA the target is R16F and the shader must encode; without it the swapchain
         * attachment does the transfer in hardware when its format is sRGB).
         */
        [[nodiscard]] bool resolve(resolve_context const& context, resolved_io& out) const override;
        void record(resolved_io const& io) override;

        /// @brief whether the pass built everything it records with (the renderer gates the post chain on this)
        [[nodiscard]] bool pipeline_ready() const noexcept;
        /// @brief the framework's generic form of the same question, so an owner holding only a chain can ask it
        ///        (a pass with nothing of its own to build keeps the interface's 	rue; see rame_pass::ready)
        [[nodiscard]] bool ready() const noexcept override {
            return this->pipeline_ready();
        }
        /// @brief the pipeline for a target in the SWAPCHAIN's format (the FXAA-off frame)
        [[nodiscard]] VkPipeline composite_pipeline() const noexcept;
        /// @brief the R16F pipeline: the bloom levels, and the LDR target FXAA will read
        [[nodiscard]] VkPipeline hdr_pipeline() const noexcept;
        /// @brief the pipeline the runner binds by default: the SWAPCHAIN variant, which `resolve` replaces on
        ///        the frames FXAA finishes (the declaration names ONE pipeline, and this pass owns both variants)
        [[nodiscard]] VkPipeline pipeline() const noexcept override;
        /**
         * @brief the R16F variant, which the BLOOM LEVELS record with
         *
         * The four bloom passes declare `post_hdr` in their behaviour and own nothing: this is the pass that owns the
         * object, and answering by NAME is what lets the chain share it without the renderer knowing whose it is
         * (see `frame_pass::named_pipeline`).
         */
        [[nodiscard]] owned_pipeline named_pipeline(std::string_view name) const noexcept override;
        /// @brief the layout every post pipeline binds its set and takes its push block through
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept override;
        /**
         * @brief the GI upsample's ON/OFF lane, which is THIS pass's parameter
         *
         * It is read by the composite alone (the spatial filter has its own criterion), so it lives here rather
         * than in the frame's settings - the rule `vulkan.frame_constants::render_settings` states.
         */
        void set_gi_upsample(bool enabled) noexcept;

        void set_frame(composite_frame const& frame) noexcept;

    private:
        static constexpr std::string_view vertex_shader_name = "post.vert.spv"; // the synthetic fullscreen triangle
        static constexpr std::string_view fragment_shader_name = "post.frag.spv";

        /// ONE name: the behaviour says which pipeline the runner binds, and the host fills that slot with the
        /// variant the frame's TARGET needs (see composite_pipeline / hdr_pipeline) - two names here would make
        /// the runner bind both, and the second would win.
        static constexpr std::array<std::string_view, 1> pipeline_names = {"post_composite"};
        inline static constexpr vulkan::pass::behaviour behaviour_ = {
            .kind = behaviour_kind::fullscreen,
            .extent = extent_rule::full, // the composite runs at the frame's resolution
            .extent_of = resource_id::none,
            .pipelines = pipeline_names,
            .resync_viewport = true, // the runner sets the viewport and scissor from io.extent
        };
        void release_owned() noexcept;
        /// @brief compose the chain's push block for this frame's target choice (see resolve)
        [[nodiscard]] bool fill_push(resolved_io& out, bool writing_ldr) const;

        VkDevice device_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        std::optional<vk_pipeline> composite_ = std::nullopt;
        std::optional<vk_pipeline> hdr_ = std::nullopt;
        /// the surface's format, cached at create: the `encode_gamma` lane is a consequence of it (and of the
        /// frame's target choice), and a session-stable device fact is exactly what a create step may cache
        VkFormat swap_chain_format_ = VK_FORMAT_UNDEFINED;
        /// the GI upsample's lane (see set_gi_upsample)
        bool gi_upsample_ = true;
        composite_frame frame_ = {};
    };

    /**
     * @brief ONE bloom level: the bright-pass prefilter (level 0) or a downsample (levels 1..3)
     *
     * Four instances of this class, one per level, and the LEVEL is what differs: the target it renders into
     * (element `level` of the `bloom` family), the resource it declares for its input transition (level
     * `level - 1`; level 0 reads the HDR target, whose transition the frame loop owns), the extent the host
     * derives from `behaviour::extent_of_element`, and the push block's `mode` lane (0 for the prefilter, 1 for
     * every downsample - the shader's bright-pass test runs on the prefilter only).
     *
     * The behaviour is a MEMBER here rather than a static constant like every other pass's, because the element
     * is part of it and four instances cannot share one.
     */
    class post_bloom_pass final : public frame_pass {
    public:
        /// @param level which level of the chain this instance is (0..3); a level outside the family is clamped
        ///        to 0 rather than left to index the declaration list out of range
        explicit post_bloom_pass(uint32_t level) noexcept;
        ~post_bloom_pass() override;

        [[nodiscard]] render_resource::pass_io const& io() const noexcept override;
        [[nodiscard]] vulkan::pass::behaviour const& behaviour() const noexcept override;
        [[nodiscard]] std::string_view feature() const noexcept override;
        void create(pass_context const& context) override;
        void on_swapchain_recreated(pass_host const& host) override;
        void record(resolved_io const& io) override;

        /// @brief which level of the bloom chain this instance is
        [[nodiscard]] uint32_t level() const noexcept;

    private:
        /// the pipeline the runner binds: the chain's R16F variant, which the COMPOSITE owns. The NAME is the
        /// composite's (`post_composite_pass::bloom_pipeline_name`) because the owner resolves it by asking ITS
        /// passes (`frame_pass::named_pipeline`) - and one shared object is the point: a copy per level would be
        /// five identical pipelines.
        static constexpr std::array<std::string_view, 1> pipeline_names = {post_composite_pass::bloom_pipeline_name};

        uint32_t level_ = 0;
        render_resource::pass_io const* io_ = nullptr;
        vulkan::pass::behaviour behaviour_ = {};
    };

} // namespace vulkan::pass
