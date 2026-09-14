// module version: 0.1.0  (independent of the app version in CMakeLists project(VERSION))

/**
 * @file vulkan/pass/pass.cppm
 * @brief What a pass IS: its declaration, the behaviour that says how to call it, and the runner that does.
 * @defgroup vulkan_pass Frame Pass Framework
 *
 * THE PROBLEM THIS REMOVES, measured rather than asserted. Adding one render pass to this renderer today costs,
 * by the inventory in `docs/runtime_split.md`: a push-constant struct, a pipeline, a pipeline layout, a
 * descriptor family, a `make_*`, an `ensure_*`, a `record_*`, a `gpu_mark_id` enumerator and its call, a
 * `render_features` field, two feature strings, a viewport-resync line, a flag reset in the constructor AND in
 * `on_swapchain_recreated`, and a destroy in the destructor. This module is the shape that replaces that list
 * with "declare the I/O, implement record, name it in a stage".
 *
 * WHERE EACH FACT LIVES, because the point of this layer is that no fact lives twice:
 *
 *  * `vulkan.render_resource` owns WHAT EXISTS and WHAT A PASS USES (pure data, ctest-tested, no device);
 *  * this module owns HOW A PASS IS CALLED (behaviour), WHAT A PASS IS GIVEN (`resolved_io`), and the ORDER a
 *    stage's passes run in (declaration order, never container order);
 *  * `vulkan.core` keeps owning every image; `vulkan.runtime` keeps owning every pipeline. Neither moves.
 *
 * THE ONE INTERFACE A PASS HAS IS `resolved_io`: the handles its own declaration asked for, indexed by its own
 * binding numbers. The runtime resolves them FROM the declaration, so a pass cannot reach a resource it did not
 * declare - which is what makes the dependency graph checkable instead of aspirational. `pass_host` is
 * deliberately NOT that interface: it is what the RUNNER talks to. Keeping the host away from passes is what
 * stops this from growing into a context object that hands out whatever the newest pass happens to want.
 *
 * WHAT IT IS NOT: not an allocator, not a barrier generator, not an ordering engine. The frame's stage order is
 * declared (it is the recording spine, and `gpu_mark_id`'s positional contract lives on it), and barriers stay
 * hand-written until the layer that derives them has its own measured step.
 */

module;

#include <cstdint>
#include <span>
#include <string_view>
#include <vulkan/vulkan.h>

export module vulkan.pass;

import vulkan.render_resource;

export namespace vulkan::pass {

    using render_resource::resource_id;

    /// forward: `pass_host` names it in a callback, and the class itself names `pass_host`
    class frame_pass;

    // =============================================================================================
    // 1. HOW A PASS IS CALLED - one small closed vocabulary, shared by every pass of that shape
    // =============================================================================================

    /// @brief where a pass's dispatch or draw extent comes from
    enum class extent_rule : uint8_t {
        full,     // the frame's extent
        half,     // half the frame's extent - the GI chain's resolution
        resource, // the extent of the resource named in `extent_of` (the probe grid is not the frame's size)
    };

    /// @brief the shape of the work: what the runner must do AROUND the pass, not what the pass computes
    enum class behaviour_kind : uint8_t {
        compute,    // a dispatch: the pass records it, the extent and the workgroup size are declared
        fullscreen, // one fullscreen triangle per target; the viewport MUST be resynced (see `resync_viewport`)
        graphics,   // a draw into a rendering instance the stage opened
        instanced,  // one draw per instance - the shadow cascades' shape
    };

    /**
     * @brief how this pass wants to be invoked
     *
     * `group_size` is a DECLARED FACT, not a convenience: it must equal the shader's `local_size_x/y/z`, and
     * that equality is exactly the kind of thing this layer exists to be able to check later against the
     * SPIR-V (the reflection parser is already in `vulkan.core.pipeline.spirv_parser`). Today the same number
     * lives in a shader and in a dispatch call, in two files, with nothing tying them together.
     */
    struct behaviour {
        behaviour_kind kind = behaviour_kind::compute;
        uint32_t group_size_x = 8;
        uint32_t group_size_y = 8;
        uint32_t group_size_z = 1;
        extent_rule extent = extent_rule::full;
        resource_id extent_of = resource_id::none; // read only when extent == resource
        /**
         * The pipelines this pass records with, BY NAME, in the order it will use them.
         *
         * NAMES RATHER THAN A BUILD REQUEST, decided deliberately: `vulkan.runtime` already owns the pipelines
         * and already keys them by name (`make_pipeline` / `set_default_pipeline` / `get_pipeline`), so a pass
         * naming what it needs is the existing mechanism rather than a new one - and it keeps the 787 lines of
         * `vulkan.pipelines` where they are, with the pipeline layouts still built there. The host resolves the
         * names into `resolved_io::pipelines`, in this order, so `pipelines[i]` is the i-th name here.
         */
        std::span<std::string_view const> pipelines = {};
        /**
         * Whether the runner must resynchronise the viewport and scissor before this pass.
         *
         * THIS FIELD EXISTS BECAUSE OF A MEASURED HAZARD: the viewport resync is a hand-maintained list of
         * pipelines in `update_pass_geometry` today, and dropping a pipeline from it makes a post pass set a
         * zero-width viewport. A behaviour kind that cannot forget it is the fix, and `fullscreen` is the kind
         * that sets it.
         */
        bool resync_viewport = false;
    };

    /// @brief the frame a pass is being recorded in: the two counters this project has confused before
    struct frame_identity {
        uint32_t image_index = 0; // per-swapchain-image resources (GI, TAA, the G-buffer)
        uint32_t slot = 0;        // per-frame-slot resources (shadow maps, the light/camera buffers)
        VkExtent2D extent = {0, 0};
    };

    /// @brief one resolved own binding: exactly one of the two handles is set, according to the binding's kind
    struct resolved_binding {
        VkImageView view = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
    };

    /**
     * @brief the SHARED sets, as they are: a pass that declared usage of one may bind it directly
     *
     * Three owners exist today (the scene set, the G-buffer set, the post set) and they are shared by
     * construction - the G-buffer set alone carries the GI chain's images, the probe's SH-2 coefficients and
     * the post pass's depth and normal. A pass DECLARES that it uses a shared binding (`set_owner`) and this
     * is the set behind it; who actually binds it is the host's business, and the runtime already binds the
     * scene set before a draw.
     */
    struct shared_sets {
        VkDescriptorSet scene = VK_NULL_HANDLE;
        VkDescriptorSet gbuffer = VK_NULL_HANDLE;
        VkDescriptorSet post = VK_NULL_HANDLE;
    };

    /**
     * @brief what a pass is given: its own declaration, resolved
     *
     * `own` is indexed by the pass's OWN BINDING NUMBER - `vulkan.render_resource`'s validator requires those
     * to be contiguous from zero, so the index IS the declaration's `binding` field and nothing is looked up
     * in the frame path.
     */
    struct resolved_io {
        frame_identity frame = {};
        /// THE command buffer is handed out per frame, at recording time, and never stored: a pass records
        /// into what it is given, which is why it holds no device state between frames.
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        /// the pass's own binding number -> the two handles a binding can be (exactly one is set)
        std::span<resolved_binding const> own = {};
        VkDescriptorSet own_set = VK_NULL_HANDLE;
        shared_sets shared = {};
        /// in the order `behaviour::pipelines` names them, one entry per name
        std::span<VkPipeline const> pipelines = {};
        /// the extent THIS pass works at: the frame's, half of it, or a resource's, per `behaviour::extent`
        VkExtent2D extent = {0, 0};
    };

    // =============================================================================================
    // 2. WHAT THE RUNNER TALKS TO - the host, filled once by the runtime
    // =============================================================================================

    /**
     * @brief the runner's interface to the renderer: callbacks plus a context, no virtuals, no allocation
     *
     * The same injection shape `vulkan.animation`'s `backend` uses (a struct of callbacks and a context, passed
     * in rather than inherited), which is why this framework depends on NEITHER `vulkan.runtime` NOR
     * `vulkan.core`: `main.cpp`'s replacement, or a test, can supply one.
     *
     * IT IS NOT FOR PASSES. A pass sees only `resolved_io`.
     */
    struct pass_host {
        void* context = nullptr;
        /// the frame being recorded
        frame_identity (*frame)(void* context) = nullptr;
        /// whether a pass's feature is active this frame (`feature()`; empty means always)
        bool (*feature_active)(void* context, std::string_view feature) = nullptr;
        /// resolve a pass's own declaration to handles; false means "this frame cannot run it"
        bool (*resolve)(void* context, frame_pass const& pass, resolved_io& out) = nullptr;
        /// the mechanical pre-record step the behaviour asks for: bind the pipelines, resync the viewport
        void (*apply_behaviour)(void* context, frame_pass const& pass, resolved_io const& io) = nullptr;
        /// open and close one stage's timing interval: the STAGE owns the mark, not the pass
        void (*mark_begin)(void* context, std::string_view stage_name) = nullptr;
        void (*mark_end)(void* context, std::string_view stage_name) = nullptr;
    };

    // =============================================================================================
    // 3. WHAT A PASS IS - the base every pass derives from, shaped like vulkan.primitive
    // =============================================================================================

    /**
     * @brief the base class of every frame pass: pure virtual, one `final` class per pass
     *
     * Modelled on `vulkan.primitive`, which has carried the renderer's dynamic dispatch since the primitive
     * work: a small pure-virtual interface, `final` derived classes, and the CONTRACT written down - there,
     * "the runtime binds the pipeline and the scene set before calling draw()"; here, "the runner validates
     * `io()`, resolves it, applies `behaviour()`, then calls `record()`".
     *
     * `feature()` returns a NAME rather than an enumerator: which features exist and who enables them is the
     * runtime's registry, not this layer's business, and a pass must be recordable-or-not without this module
     * knowing the list. The host resolves it; the cost is one string compare per pass per frame against a
     * dozen passes.
     */
    class frame_pass {
    public:
        frame_pass() = default;
        frame_pass(frame_pass const&) = delete;
        frame_pass& operator=(frame_pass const&) = delete;
        virtual ~frame_pass() = default;

        /// @brief the declaration: what it uses, in `vulkan.render_resource`'s vocabulary
        [[nodiscard]] virtual render_resource::pass_io const& io() const noexcept = 0;
        /// @brief how the runner must call it
        [[nodiscard]] virtual behaviour const& behaviour() const noexcept = 0;
        /// @brief the feature that gates it ([render] keys); empty means "always"
        [[nodiscard]] virtual std::string_view feature() const noexcept = 0;
        /// @brief build what this pass owns (pipelines, layouts) through the host; once per device generation
        virtual void create(pass_host const& host) = 0;
        /// @brief the swapchain was rebuilt, so every per-image resource this pass held is stale
        virtual void on_swapchain_recreated(pass_host const& host) = 0;
        /// @brief record into the frame, with the resources the declaration asked for already resolved
        virtual void record(resolved_io const& io) const = 0;
    };

    // =============================================================================================
    // 4. A STAGE - a group of passes the runtime records together, in declaration order
    // =============================================================================================

    /**
     * @brief one stage of the frame: the passes it runs and the marks around them
     *
     * A stage exists so that the ORDER and the MARKS stop being positional. Today the frame's order is a
     * recording spine and `gpu_mark_id` is a private nested enum whose order is a contract with 15 call sites
     * in eight subsystems - an extracted pass cannot even name that enumeration. Here the stage writes its own
     * pair around its children, so a pass says which stage it belongs to and nothing else.
     *
     * The passes are POINTERS in declaration order, never a container whose iteration order is an accident:
     * this renderer's verification rests on byte-identical captures, and a hash map's order would break it.
     */
    struct stage {
        std::string_view name = {};
        /// MUTABLE pointers: `create` and `on_swapchain_recreated` store what the pass owns into it, while
        /// `record` is const because it mutates nothing - it is handed everything it needs.
        std::span<frame_pass*> passes = {};
        bool marks = true; // a stage nested inside another's instance may not want its own pair
    };

    /**
     * @brief what the runner did, which is what its test asserts
     *
     * A report rather than a log: the contract of this layer ("create once, resolve per frame, skip an inactive
     * pass WITHOUT resolving it, mark once per stage, recreate every pass on a new generation") is only worth
     * anything if it can be checked - and checked without a device.
     */
    struct run_report {
        uint32_t created = 0;
        uint32_t recorded = 0;
        uint32_t skipped_inactive = 0;
        uint32_t skipped_unresolved = 0;
        uint32_t marked = 0;
        uint32_t recreated = 0;
        /// the name of the pass whose declaration the runner refused; empty when the stage was built
        std::string_view rejected = {};
    };

    /**
     * @brief build every pass in the stage, and REFUSE the whole stage if any declaration is invalid
     *
     * The validation is `vulkan.render_resource::validate`, which is pure data: it needs no device, so a bad
     * declaration is caught at startup on every machine rather than by a validation-layer message at submit
     * time, on the machine that happens to run that pass.
     */
    [[nodiscard]] inline run_report create_stage(stage const& st, pass_host const& host) {
        run_report report;
        for (frame_pass* const p : st.passes) {
            if (p == nullptr) {
                continue;
            }
            if (!render_resource::validate(p->io())) {
                report.rejected = p->io().name;
                return report;
            }
            p->create(host);
            ++report.created;
        }
        return report;
    }

    /**
     * @brief record one stage: resolve, apply the behaviour, record, and bracket it with the stage's mark
     *
     * THE ORDER OF THESE STEPS IS THE CONTRACT, and each has a reason:
     *  1. an INACTIVE pass is not resolved and not recorded at all - several passes document exactly that
     *     ("where any requirement is missing the pass is not even recorded"), and it is what makes a feature
     *     that is off byte-identical rather than merely invisible;
     *  2. `resolve` may still fail for a frame (a descriptor family could not be had), and that is skipped
     *     WITHOUT recording, because a pass recorded with unresolved handles is worse than a pass not running;
     *  3. the behaviour's mechanical part happens BEFORE `record()`, never inside it, which is what makes the
     *     viewport resync unforgettable;
     *  4. the marks bracket the stage, so a pass cannot mark out of order - the defect the positional
     *     `gpu_mark_id` enumeration allows today.
     */
    [[nodiscard]] inline run_report record_stage(stage const& st, pass_host const& host) {
        run_report report;
        bool const marks_open = st.marks && host.mark_begin != nullptr && host.mark_end != nullptr;
        if (marks_open) {
            host.mark_begin(host.context, st.name);
        }
        for (frame_pass* const p : st.passes) {
            if (p == nullptr) {
                continue;
            }
            if (host.feature_active != nullptr && !p->feature().empty() && !host.feature_active(host.context, p->feature())) {
                ++report.skipped_inactive;
                continue;
            }
            resolved_io io = {};
            if (host.resolve == nullptr || !host.resolve(host.context, *p, io)) {
                ++report.skipped_unresolved;
                continue;
            }
            if (host.apply_behaviour != nullptr) {
                host.apply_behaviour(host.context, *p, io);
            }
            p->record(io);
            ++report.recorded;
        }
        if (marks_open) {
            host.mark_end(host.context, st.name);
            ++report.marked;
        }
        return report;
    }

    /**
     * @brief tell every pass in the stage that the swapchain was rebuilt
     *
     * THIS FUNCTION REMOVES A KNOWN HAZARD BY CONSTRUCTION. The manual reset list in `on_swapchain_recreated`
     * retires four of the six `image_set_family` instances this renderer owns; the other two survive only
     * because `ensure()` re-detects changed views. A pass that owns a family and does not get this call keeps
     * stale descriptors, so the runner makes the call instead of the runtime remembering - and
     * `run_report::recreated` counts what it did.
     */
    [[nodiscard]] inline run_report recreate_stage(stage const& st, pass_host const& host) {
        run_report report;
        for (frame_pass* const p : st.passes) {
            if (p == nullptr) {
                continue;
            }
            p->on_swapchain_recreated(host);
            ++report.recreated;
        }
        return report;
    }

} // namespace vulkan::pass
