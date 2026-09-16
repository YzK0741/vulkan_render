// module version: 0.4.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/ssgi_trace.cppm
 * @brief The fifth real pass, and the first GI stage: one bounce of screen-space diffuse indirect light, traced
 *        at half resolution.
 * @defgroup vulkan_pass_ssgi_trace SSGI Trace Pass
 *
 * WHAT MAKES THIS PASS DIFFERENT FROM EVERY ONE BEFORE IT: it binds TWO SHARED SETS and owns nothing - no set
 * layout, no descriptor family, not one binding - and yet it is the first pass whose work cannot be described
 * without naming IMAGES. The tracer writes `gi_trace` as a storage image, samples last frame's `gi_resolve` as
 * the multi-bounce feedback, samples the probe cache's coefficient volumes, and is the frame's first reader of
 * two accumulations the denoiser has not rewritten yet this frame. Every one of those images lives in the
 * G-buffer set, whose contents its OWNER writes, so the pass can neither bind them nor describe them - which is
 * exactly the gap the plan recorded as the GI chain's blocker, and `pass_io::barrier_images` is what closed it:
 * a layout transition names an image and no descriptor, so a pass may declare the images it must move and the
 * host hands it their handles.
 *
 * WHAT IT OWNS: its pipeline layout and its compute pipeline (built from `ssgi.comp` - the shader bytes arrive
 * through `pass_context::shader`), the dispatch, the whole first-use barrier batch, the hand-off barrier, and
 * the ONE piece of per-generation state that batch needs (whether it has already seen this generation's probe
 * grid - the same shape as the TAA resolve's generation fingerprint, and reset by the same contract:
 * `on_swapchain_recreated`).
 *
 * WHAT IT DELIBERATELY DOES NOT OWN, and the list is the honest half:
 *  * whether the previous frame's accumulation may be trusted (`history_valid`) - that is the DENOISER's state,
 *    and the host reads it off that pass and hands it over;
 *  * whether the glossy lobe runs next, because that decides whether the tracer owes the denoiser its hand-off
 *    barrier NOW or after the lobe has added to the same image (both passes write `gi_trace`);
 *  * the frame's own facts - the camera (`inv_view_proj`), the projection's two depth terms, the scene bounds the
 *    probe grid is anchored to, the instance table's device address, the ray sequence, and which oracle this frame
 *    uses - which arrive through `resolved_io::constants` and this pass's frame. The push block is composed by
 *    THIS pass, from those facts and its own ray budget.
 */

module;

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vulkan/vulkan.h>

export module vulkan.pass.ssgi_trace;

import vulkan.pass;
import vulkan.render_resource;
import vulkan.core.handles; // vk_pipeline: the RAII owner of the compute pipeline this pass builds

export namespace vulkan::pass {

    /**
     * @brief what the renderer hands the tracer: the frame facts it cannot derive, and nothing else
     *
     * The images arrive through `resolved_io::barrier_images` (declared by the pass, resolved by the host) and the
     * two shared sets through `resolved_io::shared`; the camera, the knobs and the extents are the pass's own
     * parameters and the frame's constants, so this struct is exactly the frame's knowledge.
     */
    struct ssgi_trace_frame {
        /// whether the previous frame's diffuse accumulation may be trusted (the denoiser's per-image state)
        bool history_valid = false;
        /// whether the glossy lobe runs after this pass, in which case the hand-off barrier is THAT pass's
        bool specular_next = false;
        /// whether this is the first dispatch of the generation, so the probe grid needs its first-use batch
        bool probe_grid_first_use = false;
        /**
         * Which ORACLE this frame's rays use: the traced one (ray queries against the scene's structures) or the
         * marched one (depth-buffer ray marching). It decides THREE lanes of the push block at once - the ray
         * origin bias against the step count, the shader's own branch, and whether the ray sequence's divisor is
         * one or two - so the renderer resolves it once and hands over the answer.
         */
        bool traced_oracle = false;
        /// whether the probe cache is on AND has been written at least once, so its gain may be pushed non-zero
        bool probe_ready = false;
        /**
         * Whether a hit this frame is shaded from the geometry it landed on, which is HALF of what the instance
         * table's address means to this shader: with hit shading off the address is pushed as ZERO, and that zero is
         * how the shader is told to read the screen where its ray landed instead. The other half is the frame's
         * table itself (`resolved_io::constants.gi_instance_table`), which the probe cache - whose hits are in
         * world space and have no screen to fall back to - uses whether or not this is true.
         */
        bool shade_hits = false;
    };

    /**
     * @brief one bounce of screen-space diffuse indirect light, at half resolution
     *
     * The frame position is load-bearing and is the renderer's: it runs after the lighting stage (so a hit
     * samples SHADED radiance and never its own output, which would make the loop gain exceed one) and before
     * the denoiser that consumes `gi_trace`.
     */
    class ssgi_trace_pass final : public frame_pass {
    public:
        /// @brief the push block, which is also `ssgi.comp`'s. Declared here because its SHAPE is the pass's
        ///        (the host fills it in), and `static_assert`ed against the declaration.
        struct push_constants {
            glm::mat4 inv_view_proj = glm::mat4(1.0f);
            glm::vec4 params = glm::vec4(0.0f);
            glm::vec4 proj_terms = glm::vec4(0.0f);
            glm::vec4 frame_info = glm::vec4(0.0f);
            glm::vec4 probe_grid = glm::vec4(0.0f);
        };

        ssgi_trace_pass() = default;
        ~ssgi_trace_pass() override;

        [[nodiscard]] render_resource::pass_io const& io() const noexcept override;
        [[nodiscard]] vulkan::pass::behaviour const& behaviour() const noexcept override;
        [[nodiscard]] std::string_view feature() const noexcept override;
        void create(pass_context const& context) override;
        void on_swapchain_recreated(pass_host const& host) override;
        void record(resolved_io const& io) override;

        /// @brief whether the pass built what it records with (the renderer gates the GI feature on this)
        [[nodiscard]] bool pipeline_ready() const noexcept;
        /// @brief the framework's generic form of the same question, so an owner holding only a chain can ask it
        ///        (a pass with nothing of its own to build keeps the interface's `true`; see `frame_pass::ready`)
        [[nodiscard]] bool ready() const noexcept override {
            return this->pipeline_ready();
        }
        /**
         * @brief whether this generation's probe grid has had its first-use batch
         *
         * The pass's own per-generation state, and the HOST reads it now instead of keeping a second flag: the
         * batch has to happen exactly once per generation, the pass is what applies it, and the host's flag was
         * the same fact in two places - with the extra failure mode that the two could disagree after a resize.
         */
        [[nodiscard]] bool probe_grid_seen() const noexcept;
        /// @brief the pipeline the runner binds before this pass records
        [[nodiscard]] VkPipeline pipeline() const noexcept override;
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept override;

        /**
         * @brief the tracer's own ray budget: how much indirect it adds and how far its rays reach
         *
         * THE PASS'S PARAMETERS, by the rule the framework settled on: one pass reads each of them, so the pass
         * owns them and the renderer's `set_ssgi` forwards. `radius` is a FRACTION of the scene radius (so one
         * setting means the same thing on a 1.6-unit model and on Sponza), the intensity REPLACES the ambient the
         * lighting stage would otherwise use on the traced path, and `rays` x `steps` is the cost per half-res
         * pixel - the clamps live here with the values.
         */
        void set_reach(float intensity, float radius, uint32_t rays, uint32_t steps) noexcept;
        /// @brief how much of the previous frame's accumulated indirect a hit re-emits (the multi-bounce gain)
        void set_bounce(float gain) noexcept;
        /// @brief the probe cache's gain, which is what makes its contribution measurable (0 turns it off)
        void set_probe_gain(float gain) noexcept;

        void set_frame(ssgi_trace_frame const& frame) noexcept;

    private:
        static constexpr std::string_view shader_name = "ssgi.comp.spv";
        /// the dispatch's workgroup size, which must be `ssgi.comp`'s `local_size_x/y`
        static constexpr uint32_t group_size = 8;
        /**
         * The declared barrier images, by the position the DECLARATION gives them (see its comment). Named
         * here so the pass's record() reads as a sequence of intents rather than as arithmetic on indices - and
         * asserted below, so a declaration that grows or reorders cannot silently change what the pass does.
         */
        static constexpr uint32_t barrier_gi_trace = 0;
        static constexpr uint32_t barrier_gi_spec_resolve = 1;
        static constexpr uint32_t barrier_gi_resolve = 2;
        static constexpr uint32_t barrier_probe_grid_read = 3;  // elements 0..3
        static constexpr uint32_t barrier_probe_grid_write = 7; // elements 4..7
        static constexpr uint32_t barrier_probe_surface = 11;
        static_assert(barrier_probe_surface + 1 == render_resource::ssgi_trace_barriers.size(),
                      "the tracer's barrier slots must match the declaration it indexes");

        static constexpr std::array<std::string_view, 1> pipeline_names = {"ssgi"};
        inline static constexpr vulkan::pass::behaviour behaviour_ = {
            .kind = behaviour_kind::compute,
            .group_size_x = group_size,
            .group_size_y = group_size,
            .group_size_z = 1,
            .extent = extent_rule::half, // the GI chain's resolution: half the frame, in both axes
            .extent_of = resource_id::none,
            .pipelines = pipeline_names,
            .resync_viewport = false, // a compute dispatch has no viewport
        };
        void release_owned() noexcept;

        VkDevice device_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pipeline_ = std::nullopt;
        /// whether this generation's probe grid has had its first-use batch (see on_swapchain_recreated)
        bool probe_grid_seen_ = false;
        /// the tracer's own ray budget, clamped where it is set (see set_reach / set_bounce / set_probe_gain)
        float intensity_ = 0.7f;
        float radius_ = 3.0f;
        uint32_t rays_ = 2;
        uint32_t steps_ = 6;
        float bounce_ = 0.0f;
        float probe_gain_ = 1.0f;
        ssgi_trace_frame frame_ = {};
    };

    static_assert(sizeof(ssgi_trace_pass::push_constants) == render_resource::ssgi_trace_io.push->size,
                  "the tracer's declared push block must be the size of the struct the renderer composes");

} // namespace vulkan::pass
