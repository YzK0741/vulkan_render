// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/character_forward.cppm
 * @brief The TOON CHARACTER stage: the same leaves the scene pass drew, shaded a SECOND time and
 *        written OVER the deferred result.
 * @defgroup vulkan_pass_character_forward Character Forward Pass
 *
 * WHY IT IS ITS OWN PASS, and the reference this follows. The port's reference implementation
 * (DanbaidongRP's PBRToon material family, see _toon_ref/PORT_SPEC.md) draws its character materials in a
 * pass named `CharacterForward` carrying `ZWrite Off` + `ZTest Equal`, invoked after the deferred lighting
 * stage.
 * The two states together are the mechanism: the pass lands on exactly the surface the G-buffer pass
 * recorded, so it REPLACES the lit pixel instead of layering over it - no mask, no stencil, and no
 * double lighting. It cannot be a step inside the lighting instance, because that instance samples the
 * depth this pass needs as an attachment.
 *
 * WHERE IT SITS: after the lighting stage (and after the transparent pass, which composites over the
 * lit frame) and before the resolve. A toon character drawn before the lighting stage would be lit by
 * it, which is the single thing this pass exists to prevent.
 *
 * WHAT IT OWNS: the two targets entered with LOAD, the two barriers that hand the depth between
 * "sampled" and "attachment" layouts (it is what takes it out of SHADER_READ and what puts it back), and
 * the raster state the stage needs - depth compare EQUAL comes from the PIPELINE
 * (`core::make_character_forward_pipeline`); depth WRITE off is a dynamic state this pass records and
 * then LOCKS, because every leaf's draw() otherwise turns it back on (see
 * render_environment::depth_write_locked).
 *
 * WHAT IT DOES NOT OWN: the geometry. Its leaves are the SCENE's, re-drawn - the renderer hands over the
 * same list the scene pass drew, which is what makes the two agree about which surfaces exist.
 *
 * IT IS BEHIND A FEATURE ("character_forward") THAT DEFAULTS OFF, so a frame from a scene with no toon
 * character is bit-for-bit what it was before this pass existed - the project's acceptance rule for a
 * behaviour-visible addition.
 */

module;

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vulkan/vulkan.h>

export module vulkan.pass.character_forward;

import vulkan.pass;
import vulkan.render_resource;
import vulkan.render_resource.shared;
import vulkan.constant_init;
import vulkan.primitive;
import vulkan.render_environment;

export namespace vulkan::pass {

    /// @brief what the renderer hands the character-forward pass: the leaves, the draw state and the formats
    struct character_forward_frame {
        /// the OPAQUE leaves - the same set the scene pass drew, because this pass re-shades those surfaces
        std::span<primitive const* const> leaves = {};
        /**
         * Draw state for this session, built by the renderer.
         *
         * A CALLBACK for the same reason the scene and transparent frames' is (the pipeline registry is the
         * renderer's), and the pass then OVERRIDES two things on what it gets back: `default_name` (so a
         * default-semantics leaf binds the character pipeline rather than `pbr`) and the locked depth-write
         * state. Both are fields of the environment rather than callback arguments, which is what lets this
         * pass state its own raster rules without the renderer having to know them.
         */
        render_environment (*make_environment)(void* owner, VkCommandBuffer command_buffer, bool gbuffer) = nullptr;
        void* owner = nullptr;
        /**
         * The pipeline name every leaf of this session binds.
         *
         * THE RENDERER FILLS IT rather than the pass hardcoding it, because the name is the runtime's key
         * into its own registry - the pass would be asserting that some other module registered a pipeline
         * under a string it chose. Empty means the session cannot bind anything and the pass draws nothing.
         */
        std::string_view pipeline_name = {};
        /// the two declared targets' formats and the extent (the pass opens its own instance)
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        VkFormat depth_format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent = {0, 0};
    };

    /**
     * @brief the toon character stage: the scene's own opaque leaves, re-shaded over the lit frame
     *
     * @note the pass draws nothing when its leaf set is empty or the session has no pipeline name. That is
     *       not an optimisation: a frame with no character has no toon surface, and skipping it keeps such
     *       a frame bit-for-bit what it was.
     */
    class character_forward_pass final : public frame_pass {
    public:
        character_forward_pass() = default;
        ~character_forward_pass() override = default;

        [[nodiscard]] render_resource::pass_io const& io() const noexcept override;
        [[nodiscard]] vulkan::pass::behaviour const& behaviour() const noexcept override;
        [[nodiscard]] std::string_view feature() const noexcept override;
        void create(pass_context const& context) override;
        void on_swapchain_recreated(pass_host const& host) override;
        void record(resolved_io const& io) override;

        void set_frame(character_forward_frame const& frame) noexcept;

    private:
        static constexpr std::array<std::string_view, 0> pipeline_names = {}; // the leaves name their pipelines
        inline static constexpr vulkan::pass::behaviour behaviour_ = {
            .kind = behaviour_kind::graphics,
            .extent = extent_rule::full,
            .extent_of = resource_id::none,
            .pipelines = pipeline_names,
            .resync_viewport = false, // the character pipelines carry their own stored viewport
        };

        character_forward_frame frame_ = {};
    };

} // namespace vulkan::pass
