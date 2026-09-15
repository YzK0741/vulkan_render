// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/rt_shadow.cppm
 * @brief The ninth real pass, and the first one that is NOT part of the GI chain: the ray-traced sun shadow.
 * @defgroup vulkan_pass_rt_shadow Ray-Traced Shadow Pass
 *
 * WHAT IT OWNS: its pipeline layout and its compute pipeline (built at create time from its own declaration's
 * push-block size and the two shared set layouts its owner hands over); the frame's recording - the two barriers
 * around the visibility image it rewrites, the two shared sets it binds, the push block's values (composed by the
 * renderer) and the full-resolution dispatch; and the one-shot log that says how many rays a frame traces, which
 * used to live in the runtime as a bool next to the pipeline.
 *
 * WHAT IT DOES NOT OWN, in the same shape as the tracer and the spatial filter: no descriptor set of its own.
 * The camera, the light UBO and the top level structure are the shared scene set's, the surface each ray starts
 * from is the shared G-buffer set's, and the visibility image it writes lives in the G-buffer set too - so the
 * only handle it needs is the IMAGE it transitions, and that arrives through `pass_io::barrier_images`.
 *
 * WHY IT IS A PASS RATHER THAN A `record_*` FUNCTION: its position is an ordering constraint - after the G-buffer
 * pass (whose depth and normal the rays start from) and before the lighting stage (which multiplies the sun term
 * by its result), and running it earlier would mean starting rays from the PREVIOUS frame's surface. That
 * position is now the stage's, and the frame loop's off path (the transition the lighting stage's descriptor
 * needs on a frame where this pass does not run) is what the renderer keeps - it is about the STAGE, not the
 * pass's own work.
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

export module vulkan.pass.rt_shadow;

import vulkan.pass;
import vulkan.render_resource;
import vulkan.core.handles; // vk_pipeline: the RAII owner of the compute pipeline this pass builds

export namespace vulkan::pass {

    /**
     * @brief one ray per pixel against the scene's acceleration structures, terminated on the first hit
     *
     * Optional: without it (or without a device that has ray queries) the cascaded shadow maps keep running, which
     * is why the renderer's `light_state.rt_shadows` lane - the flag the lighting stage reads - is composed from
     * this pass's own `pipeline_ready()`.
     */
    class rt_shadow_pass final : public frame_pass {
    public:
        /// @brief the push block, which is also `rt_shadow.comp`'s
        struct push_constants {
            glm::mat4 inv_view_proj = glm::mat4(1.0f); // clip -> world, the block the lighting stage uses
            // x = ray tmin, y = absolute normal-offset floor, z = relative offset scale (per unit of
            // distance from the camera), w = unused
            glm::vec4 params = glm::vec4(0.01f, 0.002f, 0.0015f, 0.0f);
        };

        rt_shadow_pass() = default;
        ~rt_shadow_pass() override;

        [[nodiscard]] render_resource::pass_io const& io() const noexcept override;
        [[nodiscard]] vulkan::pass::behaviour const& behaviour() const noexcept override;
        [[nodiscard]] std::string_view feature() const noexcept override;
        void create(pass_context const& context) override;
        void on_swapchain_recreated(pass_host const& host) override;
        void record(resolved_io const& io) override;

        /// @brief whether the pass built what it records with (the renderer gates the ray-traced path on this)
        [[nodiscard]] bool pipeline_ready() const noexcept;
        /// @brief the pipeline the runner binds before this pass records
        [[nodiscard]] VkPipeline pipeline() const noexcept;
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept;

    private:
        static constexpr std::string_view shader_name = "rt_shadow.comp.spv";
        static constexpr uint32_t group_size = 8; // `rt_shadow.comp`'s local_size_x/y
        /// the one declared barrier image, by the position the declaration gives it
        static constexpr uint32_t barrier_visibility = 0;
        static_assert(barrier_visibility + 1 == render_resource::rt_shadow_barriers.size(),
                      "the shadow pass's barrier slots must match the declaration it indexes");

        static constexpr std::array<std::string_view, 1> pipeline_names = {"rt_shadow"};
        inline static constexpr vulkan::pass::behaviour behaviour_ = {
            .kind = behaviour_kind::compute,
            .group_size_x = group_size,
            .group_size_y = group_size,
            .group_size_z = 1,
            .extent = extent_rule::full, // one ray per PIXEL: the visibility image is the frame's size
            .extent_of = resource_id::none,
            .pipelines = pipeline_names,
            .resync_viewport = false,
        };
        void release_owned() noexcept;

        VkDevice device_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        std::optional<vk_pipeline> pipeline_ = std::nullopt;
        /// whether the "tracing WxH rays per frame" line has been logged (it used to be the runtime's flag)
        bool logged_ = false;
    };

    static_assert(sizeof(rt_shadow_pass::push_constants) == render_resource::rt_shadow_io.push->size,
                  "the shadow pass's declared push block must be the size of the struct the renderer composes");

} // namespace vulkan::pass
