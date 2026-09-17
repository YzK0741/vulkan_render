// module version: 0.2.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/taa.cppm
 * @brief The SECOND real pass, and the first GRAPHICS one: the temporal anti-aliasing resolve.
 * @defgroup vulkan_pass_taa Temporal Anti-Aliasing Pass
 *
 * WHY THIS PASS IS THE INTERESTING ONE: a compute pass proved the shape first, and this one
 * proves it for the other half of the frame. A fullscreen pass owns a render TARGET (an attachment is not a
 * descriptor: it is declared as a `render_target` and bound by a rendering instance), it opens that instance
 * itself (the load op is its knowledge, not the runner's), and the two things it must not be able to forget -
 * the pipeline being bound and the viewport being set - are done FOR it by the runner before `record`.
 *
 * WHAT IT OWNS: its set layout and the four per-swapchain-image inputs it names (this frame's colour, the
 * reprojected history, the motion vectors, the depth the disocclusion guard reads), its pipeline layout, its
 * pipeline, its per-image descriptor family, its push block's SHAPE, and the two pieces of state that say what
 * its history holds: whether each image has one, and which views identified the target generation its family
 * was built against.
 *
 * WHAT IT DELIBERATELY DOES NOT OWN, and this is the second discovery this extraction produced: TWO LINES OF
 * ITS OWN SEQUENCE BELONG TO OTHER PASSES. The resolve transitions the G-buffer depth (a transition the
 * G-buffer pass's per-image flag decides) and clears the flag that says the motion vectors have been handed
 * to a fragment sampler (which stops a later pass from transitioning them again). Both are shared per-image
 * bookkeeping, both are the barrier/order stage's job in the long run, and neither is expressible here while a
 * pass can only declare its OWN bindings - so they stay with the host, which is the one that knows the flags.
 * The order the host has to preserve is recorded at the call site.
 *
 * THE GENERATION FINGERPRINT is the one piece of this pass that is not a straight move. A per-image descriptor
 * family fingerprints "the thing my sets point at", and for a per-image resource the CURRENT image's view is
 * exactly the wrong fingerprint: it changes every frame, which would make the family rewrite every set every
 * frame - and rewriting a set a pending frame names is a validation error. The generation-level identity is
 * what the fingerprint needs, and the pass already receives the signal that invalidates it: the runner calls
 * `on_swapchain_recreated` for every pass in a stage (the contract that replaced the runtime's hand-kept reset
 * list). So the pass caches the first frame's views as its generation stamp and drops them when told.
 */

module;

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <vulkan/vulkan.h>

export module vulkan.pass.taa;

import vulkan.pass;
import vulkan.render_resource;
import vulkan.render_resource.shared;
import vulkan.bindings;
import vulkan.constant_init;
import vulkan.core.handles; // vk_pipeline: the RAII owner of the pipeline this pass builds

export namespace vulkan::pass {

    /**
     * @brief the temporal resolve: blend this frame's colour with the reprojected history, then copy the
     *        result into the history image the next frame for this swapchain image will read
     *
     * A fullscreen triangle writing the frame's HDR target. The blend weights come from the renderer (they are
     * a config knob), the reprojection comes from the motion vectors the G-buffer pass wrote, and the history
     * is the image this same pass filled on an earlier frame for the SAME swapchain image - remembered per
     * image on purpose, because with several images in rotation "the previous frame's camera" is not what that
     * image's history was rendered with.
     */
    class taa_pass final : public frame_pass {
    public:
        /**
         * @brief the pass's push block, which is also the fragment shader's
         *
         * Eight floats in the order taa.frag declares them. EVERY value in it is now the pass's own: the blend
         * weights are its parameters (see set_blend), the texel size comes from the extent it was resolved at,
         * the two depth terms come from the frame's constants, and `history_valid` is the state it maintains per
         * image. It used to be composed by the renderer and handed over as raw bytes; the block's shape is still
         * the pass's, `static_assert`ed against the declaration's size.
         */
        struct push_constants {
            float history_valid = 0.0f; // 1 = trust the history, 0 = first frame for this image
            float blend_static = 0.9f;  // history weight for a static pixel
            float blend_min = 0.5f;     // history weight floor under motion
            float texel_size_x = 0.0f;  // 1 / target width
            float texel_size_y = 0.0f;  // 1 / target height
            float depth_scale = 0.0f;   // projection[2][2]: the depth-linearization term
            float depth_offset = 0.0f;  // projection[3][2]
            float unused = 0.0f;
        };

        taa_pass() = default;
        ~taa_pass() override;

        [[nodiscard]] render_resource::pass_io const& io() const noexcept override;
        [[nodiscard]] vulkan::pass::behaviour const& behaviour() const noexcept override;
        [[nodiscard]] std::string_view feature() const noexcept override;
        void create(pass_context const& context) override;
        void on_swapchain_recreated(pass_host const& host) override;
        void record(resolved_io const& io) override;

        /// @brief whether the pass built everything it records with (the renderer gates the feature on this)
        [[nodiscard]] bool pipeline_ready() const noexcept;
        /// @brief the framework's generic form of the same question, so an owner holding only a chain can ask it
        ///        (a pass with nothing of its own to build keeps the interface's `true`; see `frame_pass::ready`)
        [[nodiscard]] bool ready() const noexcept override {
            return this->pipeline_ready();
        }
        /// @brief the pipeline the runner binds before this pass records
        [[nodiscard]] VkPipeline pipeline() const noexcept override;
        /// @brief the layout that pipeline binds its set and takes its push constants through
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept override;
        /**
         * @brief whether the last record actually resolved and wrote a new history
         *
         * The renderer's `image_view_proj` bookkeeping - the matrix the NEXT frame's motion vectors are
         * computed against - keys on this rather than on "the stage ran": a resolve that bailed out (no
         * descriptor set) must not claim a history it did not write.
         */
        [[nodiscard]] bool wrote_history() const noexcept;
        /**
         * @brief the two blend weights, which are THIS PASS's parameters - and their clamps, which are its too
         *
         * They used to live in the renderer as two members its `set_taa` cached and its resolver copied into the
         * push block. Now the renderer's public setter forwards them here, the pass clamps them once (static into
         * [0, 0.99]; the floor into [0, static]), and the values exist in exactly one place - the same rule the
         * block's other lanes already followed.
         * @param static_weight the history weight for a pixel that did not move
         * @param min_weight the history weight floor under motion (lower = less ghosting)
         */
        void set_blend(float static_weight, float min_weight) noexcept;
        /// @brief forget every image's history: the off -> on edge, when blending would resume against
        ///        frames that were never resolved
        void reset_history() noexcept;

    private:
        /// the four inputs the resolve reads, which the declaration numbers contiguously from zero
        static constexpr uint32_t own_binding_count = 4;
        static constexpr std::string_view vertex_shader_name = "post.vert.spv"; // the synthetic fullscreen triangle
        static constexpr std::string_view fragment_shader_name = "taa.frag.spv";

        static constexpr std::array<std::string_view, 1> pipeline_names = {"taa"};
        inline static constexpr vulkan::pass::behaviour behaviour_ = {
            .kind = behaviour_kind::fullscreen,
            .extent = extent_rule::full, // the resolve runs at the frame's resolution
            .extent_of = resource_id::none,
            .pipelines = pipeline_names,
            .resync_viewport = true, // the runner sets the viewport and scissor: the hazard this field exists for
        };
        void release_owned() noexcept;

        VkDevice device_ = VK_NULL_HANDLE;
        render_resource::shared::sampler_set samplers_ = {};
        VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pipeline_ = std::nullopt;
        bindings::image_set_family family_ = {};
        /// one flag per swapchain image: whether that image's history holds a resolved frame
        std::vector<bool> history_valid_ = {};
        /// the views that identified the target generation the family was built for, and whether they are set
        std::array<VkImageView, own_binding_count> generation_views_ = {};
        bool generation_views_valid_ = false;
        /// whether the last record wrote the history (see wrote_history)
        bool wrote_history_ = false;
        /// the two blend weights: this pass's own parameters (see set_blend), defaulted to the renderer's own
        /// historical defaults so a session that never calls `set_taa` resolves exactly as it did before
        float blend_static_ = 0.9f;
        float blend_min_ = 0.5f;
    };

    /// THE DECLARATION'S NUMBER AND THE PASS'S STRUCT CANNOT DRIFT: the declared push block is what the
    /// pipeline layout's range is built from and what the host composes into.
    static_assert(sizeof(taa_pass::push_constants) == render_resource::taa_io.push->size,
                  "the TAA resolve's declared push block must be the size of the struct the pass pushes");

} // namespace vulkan::pass
