// module version: 0.2.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/rt_shadow.cppm
 * @brief The ninth real pass, and the only one that traces outside the chain: the ray-traced sun shadow.
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
 * OPEN DEFECT, MEASURED, and it is why the ray-query form is still the one this renderer ships: the pipeline
 * path OVER-OCCLUDES. With a scenario that pins `rt_shadows = true`, the ray-tracing build's frame is darker
 * than the ray-query build's on 5.0% of pixels and BRIGHTER on none (model-region mean 61.2 against 76.6; the
 * ray-query frame matches the shipped cascaded reference to 0.03), so the sun term is being killed where the
 * query left it alone. What the probes established rather than guessed:
 *
 *  - the push constants ARRIVE (writing `pc.params.x * 100` as the visibility gives a fully lit frame, 78.5),
 *    `world_pos` reconstructs correctly (writing its Y as the visibility gives a smooth gradient over the
 *    model with the expected sign), and the raygen really writes the image (a constant 1.0 lights the frame);
 *  - it is NOT a mirrored write (comparing the frame against vertically, horizontally and 180-degree flipped
 *    references: as-is is by far the closest);
 *  - the hit shader's distance, written out as the visibility, shows ~20% of model pixels hit at t ~ 0 (a
 *    self-hit) and the rest at t ~ 1-4, i.e. the traversal finds geometry the query does not;
 *  - THE PIPELINE'S PLUMBING IS PROVEN GOOD, which is what makes the difference a TRAVERSAL one: a ray query
 *    run INSIDE this pass's raygen (instead of the trace) reproduces the compute build's ray-query frame to
 *    mean|d| = 0.0001 - the SBT, the entry points, the descriptors, the barriers and the lighting stage's read
 *    are all exercised by that and are all right. Two traversals of the SAME ray, built from the same origin,
 *    tmin, direction and tmax in one invocation, disagree on ~46% of the model (46.5% of its pixels land in
 *    the 'blocked' cluster against 46.3% in the 'lit' one).
 *
 * So the next experiment is to encode the QUERY's hit distance and the TRACE's hit distance for the same pixel
 * and compare them: a trace that reports plausible distances where the query reports a miss means the two are
 * traversing different things (the structure the raygen's descriptor names is the same binding, so the
 * difference would have to be in how the traversal reads its arguments), while nonsense distances would point
 * at the SBT/arguments instead. (A first hypothesis - the barriers naming COMPUTE_SHADER while the writer is
 * now the ray-tracing stage - was fixed here and changed NOTHING: the frame stayed byte-identical.)
 *
 * The SBT itself is fine (group 0 raygen / 1 miss / 2 hit, the instance's SBT record offset is 0, one geometry
 * per BLAS, `VK_GEOMETRY_OPAQUE_BIT_KHR` set, mask 0xFF, facing-cull disabled), and validation is clean. So the
 * remaining suspects are the TRAVERSAL SEMANTICS of a traced ray versus a query - and the next diagnostic is to
 * write the hit t out as the visibility on BOTH builds and diff the two, which separates "the same hits at the
 * same distances" from "different hits" before anything else is changed.
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
        /// @brief the framework's generic form of the same question, so an owner holding only a chain can ask it
        ///        (a pass with nothing of its own to build keeps the interface's `true`; see `frame_pass::ready`)
        [[nodiscard]] bool ready() const noexcept override {
            return this->pipeline_ready();
        }
        /// @brief the pipeline the runner binds before this pass records
        [[nodiscard]] VkPipeline pipeline() const noexcept override;
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept override;

    private:
        // THREE stages, and the sbt order the builder created them in: raygen, miss, hit. The regions below
        // follow that order, which is why the builder returns the group count with the pipeline.
        static constexpr std::string_view raygen_name = "rt_shadow.rgen.spv";
        static constexpr std::string_view closest_hit_name = "rt_shadow.rchit.spv";
        static constexpr std::string_view miss_name = "rt_shadow.rmiss.spv";
        static constexpr uint32_t group_size = 8; // unused by a traceRays launch (the launch dims ARE the extent)
        /// the one declared barrier image, by the position the declaration gives it
        static constexpr uint32_t barrier_visibility = 0;
        static_assert(barrier_visibility + 1 == render_resource::rt_shadow_barriers.size(),
                      "the shadow pass's barrier slots must match the declaration it indexes");

        static constexpr std::array<std::string_view, 1> pipeline_names = {"rt_shadow"};
        inline static constexpr vulkan::pass::behaviour behaviour_ = {
            .kind = behaviour_kind::ray_tracing,
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
        // The shader binding table's three regions, filled at create time from the pipeline's group handles. The
        // BUFFER is the owner's (see pass_context::create_upload_buffer); what the pass keeps is where each
        // region starts, which is the per-pipeline part.
        /// the traceRays entry point, loaded through vkGetDeviceProcAddr at create time (an extension command
        /// is not exported by the loader's import library - see the acceleration-structure module's note)
        PFN_vkCmdTraceRaysKHR trace_rays_ = nullptr;
        VkStridedDeviceAddressRegionKHR raygen_region_ = {};
        VkStridedDeviceAddressRegionKHR miss_region_ = {};
        VkStridedDeviceAddressRegionKHR hit_region_ = {};
        /// The callable region, which this pipeline has no shaders for - and which must still be a VALID
        /// pointer to an all-zero region: `vkCmdTraceRaysKHR` dereferences it, so passing nullptr is a
        /// validation error and (measured) a driver access violation rather than "no callables".
        VkStridedDeviceAddressRegionKHR callable_region_ = {};
        /// whether the "tracing WxH rays per frame" line has been logged (it used to be the runtime's flag)
        bool logged_ = false;
    };

    static_assert(sizeof(rt_shadow_pass::push_constants) == render_resource::rt_shadow_io.push->size,
                  "the shadow pass's declared push block must be the size of the struct the renderer composes");

} // namespace vulkan::pass
