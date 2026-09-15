// module version: 0.9.0  (independent of the app version in CMakeLists project(VERSION))

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

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vulkan/vulkan.h>

export module vulkan.pass;

import vulkan.render_resource;
import vulkan.render_resource.shared;
import vulkan.core.handles; // vk_descriptor_set: the RAII set `pass_context::descriptor_set` hands over

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
        /**
         * The pass sizes its OWN work and `resolved_io::extent` stays empty.
         *
         * Added for the clustered-light sort, whose dispatch is one-dimensional over `tiles_x * tiles_y *
         * slices` - a number derived from the frame's extent but equal to neither it nor half of it, so all
         * three rules above would have been a claim the host could not honour. What such a pass works at is its
         * own data, and the host hands that over in the pass's frame (see `scene_frame` for the same split).
         */
        none,
    };

    /// @brief the shape of the work: what the runner must do AROUND the pass, not what the pass computes
    enum class behaviour_kind : uint8_t {
        compute,    // a dispatch: the pass records it, the extent and the workgroup size are declared
        fullscreen, // one fullscreen triangle per target
        graphics,   // a draw per piece of scene content (a leaf, a light, a caster)
        instanced,  // one draw per instance - the shadow cascades' shape
        // THE THREE GRAPHICS KINDS ALL MEAN THE SAME THING TO THE RUNNER, and that is the point this layer
        // reached: the PASS opens the rendering instance over the targets it declared (the load op and the
        // clear value are its knowledge, not the runner's), and the runner's half is the part a pass cannot
        // forget - binding the pipeline and, for a pass that asked, setting the viewport and scissor from the
        // extent its declaration produced. What differs between the three is how many draws the pass issues
        // and from where its draw list comes, which is the pass's business and no other layer's.
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
         * Whether the runner must set the viewport and scissor from `resolved_io::extent` before this pass.
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
        /// how many swapchain images THIS generation has, which is not the same number as `image_index` and
        /// is what a pass that owns a per-image descriptor family sizes it from. The probe cache needs it,
        /// and it is the kind of fact that used to be reachable only from inside `vulkan.runtime`
        uint32_t image_count = 0;
        VkExtent2D extent = {0, 0};
    };

    /// @brief one resolved own binding: the handles this binding's resource actually is
    struct resolved_binding {
        VkImageView view = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        /**
         * The image BEHIND @c view, because a descriptor takes a view and a BARRIER takes an image.
         *
         * This field exists because the first real pass needed it: the probe cache transitions and clears
         * its own nine images (same-layout storage barriers for the propagation, a clear when the global
         * lighting changed), and a pass that has only views cannot name them. It is resolved from the same
         * declaration element as the view, so a pass still reaches nothing it did not declare.
         */
        VkImage image = VK_NULL_HANDLE;
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

    /// @brief how many own bindings one pass may resolve (the probe cache's nine are the most today)
    inline constexpr uint32_t max_own_bindings = 16;
    /// @brief how many pipelines one pass may name (the post chain's five are the most today)
    inline constexpr uint32_t max_pass_pipelines = 8;
    /// @brief how many images one pass may render into (one fullscreen pass has one; the deferred scene has five)
    inline constexpr uint32_t max_render_targets = 8;
    /// @brief how many images one pass may declare for its own transitions (the SSGI tracer's twelve are the most)
    inline constexpr uint32_t max_barrier_images = 16;
    /// @brief how many BUFFERS one pass may declare for its own ordering (the cluster sort's two are the only ones today)
    inline constexpr uint32_t max_barrier_buffers = 8;
    /// @brief the largest push block a pass may declare: the 128 bytes Vulkan guarantees
    inline constexpr uint32_t max_push_bytes = 128;

    /**
     * @brief what a pass is given: its own declaration, resolved
     *
     * `own` is indexed by the pass's OWN BINDING NUMBER - `vulkan.render_resource`'s validator requires those
     * to be contiguous from zero, so the index IS the declaration's `binding` field and nothing is looked up
     * in the frame path.
     *
     * THE STORAGE LIVES HERE, in fixed arrays the host fills and the spans view. That shape is deliberate:
     * the runner creates this struct per pass per frame and the pass records from it immediately, so one
     * struct owns everything the pass reads and there is no second place for a handle to live (and no
     * lifetime for a host to get wrong).
     */
    struct resolved_io {
        frame_identity frame = {};
        /// THE command buffer is handed out per frame, at recording time, and never stored: a pass records
        /// into what it is given, which is why it holds no device state between frames.
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        /// the storage `own` views
        std::array<resolved_binding, max_own_bindings> own_storage = {};
        /// the pass's own binding number -> the handles that binding's resource is (exactly one is set)
        std::span<resolved_binding const> own = {};
        /**
         * The same own bindings, ONE VIEW PER SWAPCHAIN IMAGE: `own_per_image[k][image]` is the view binding @c k
         * has for swapchain image @c image (each span is `frame.image_count` long, or empty where the host filled
         * nothing).
         *
         * WHY A PASS NEEDS THIS, measured rather than anticipated: a pass that owns a per-image descriptor family
         * has to write EACH IMAGE's views into THAT IMAGE's set - `image_set_family::ensure` hands its write
         * callback an image index for exactly that reason - and `own` only carries the CURRENT frame's handles.
         * The host-written families reach into the core for this (`vulkan_core.gi_images[i]` and friends); a pass
         * cannot, which is what blocked the GI denoiser's extraction twice (see docs/pass_chain_plan.md: the
         * second failure was a descriptor pointing at another image, in another layout).
         *
         * The FIRST entry is also what a per-generation fingerprint wants: `own_per_image[k][0]` is stable for as
         * long as the target generation lives, while `own[k].view` changes with every frame's image.
         */
        std::array<std::span<VkImageView const>, max_own_bindings> own_per_image = {};
        VkDescriptorSet own_set = VK_NULL_HANDLE;
        shared_sets shared = {};
        /**
         * The storage `targets` views: the images this pass RENDERS INTO, in the order its declaration names
         * them, each with the view a rendering instance takes and the image a barrier takes.
         *
         * A pass declares a target because an attachment is a use that cannot be a descriptor (see
         * `render_target`): it is bound by `vkCmdBeginRendering`, not by a set. The pass that renders into it
         * OPENS the rendering instance - the load op and the clear value are its business, since only it knows
         * whether the old contents matter - and the runner's job is to have the pipeline bound and the
         * viewport set before it does.
         */
        std::array<resolved_binding, max_render_targets> target_storage = {};
        std::span<resolved_binding const> targets = {};
        /**
         * The images this pass declared as BARRIER IMAGES (`pass_io::barrier_images`), in the declaration's
         * order - the handles it may move between layouts but never bind as a descriptor.
         *
         * This is a THIRD channel rather than a corner of `own`, and the distinction is the point: `own` is
         * indexed by the pass's own binding numbers and its sets are generated from it, while these resources
         * live in sets its OWNER writes (the G-buffer set) and the pass only needs their IMAGES. A pass that
         * wants one of these as a descriptor says so with a `shared` binding; a pass that only needs to
         * transition it declares it here.
         */
        std::array<resolved_binding, max_barrier_images> barrier_storage = {};
        std::span<resolved_binding const> barrier_images = {};
        /**
         * The BUFFERS this pass declared as BARRIER BUFFERS (`pass_io::barrier_buffers`), in the declaration's
         * order - the handles it may order around with a buffer memory barrier but never bind itself.
         *
         * The same argument as the images above, one resource class over, and it was measured rather than
         * assumed: the clustered-light sort writes two buffers that live in the SHARED scene set, so its
         * declaration names no binding for either - and without their handles it could not place the barrier
         * that makes its writes visible to the fragment stages reading them later in the same submission. Each
         * entry carries the BUFFER (the `view` and `image` lanes stay null: a barrier takes a buffer, and this
         * resource has no view).
         */
        std::array<resolved_binding, max_barrier_buffers> barrier_buffer_storage = {};
        std::span<resolved_binding const> barrier_buffers = {};
        /// the storage `pipelines` views
        std::array<VkPipeline, max_pass_pipelines> pipeline_storage = {};
        /// in the order `behaviour::pipelines` names them, one entry per name
        std::span<VkPipeline const> pipelines = {};
        /**
         * The layout of the pipelines this pass names - ONE, not one per pipeline, because that is what this
         * renderer's passes have: a pass builds one `VkPipelineLayout` and every pipeline it records with
         * (post's five, TAA's two, the probe cache's one) is created from it.
         *
         * WHY A PASS NEEDS IT AT ALL, which the first real pass made unavoidable: a pass that records its own
         * dispatches pushes its own constants and binds its own sets, and both calls take the layout. The
         * alternative - the host pushing on the pass's behalf - would mean the host knowing every pass's push
         * block, which is the same fact in two places.
         */
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        /**
         * The push block the HOST composed for this pass this frame, as raw bytes.
         *
         * Raw, because the framework has no pass's type and will not learn one: the declaration's `push`
         * block is the size contract, and the pass reads the bytes as the struct it declared. The host
         * composes it because the values in it - the scene's bounds, an instance table's device address, the
         * global light direction - are the renderer's, not the pass's; what the pass owns is the block's
         * SHAPE and the fields it changes per dispatch (the probe cache's mode lane).
         */
        std::array<std::byte, max_push_bytes> push_storage = {};
        std::span<std::byte const> push = {};
        /// the extent THIS pass works at: the frame's, half of it, or a resource's, per `behaviour::extent` -
        /// and EMPTY for a pass that declared `extent_rule::none`, which sizes its own work from its frame
        VkExtent2D extent = {0, 0};
    };

    // =============================================================================================
    // 2. WHAT A PASS IS GIVEN (create) AND WHAT THE RUNNER TALKS TO (record)
    //
    // TWO STRUCTS, and the split is a decision rather than bookkeeping: they have different OWNERS and
    // different LIFETIMES. `pass_context` answers "build what you own" and can be filled by anyone who has a
    // device and the shared facts (the runtime is one such owner, not the only one) - which is what lets a
    // pass exist outside this renderer. `pass_host` answers "run this frame" and only a frame loop can fill it.
    // =============================================================================================

    /**
     * @brief what a pass is given to BUILD what it owns; fillable by any owner, not only a runtime
     *
     * WHY IT IS SEPARATE FROM `pass_host`: a pass's create step needs things the frame loop does not have and
     * a frame loop needs things a create step does not. Merging them produced one struct that grew with every
     * pass - the shape this layer exists to avoid - and it made "who may create a pass" the same question as
     * "who may run a frame", which is not true: an editor, a test or another renderer's main() can build these
     * three facts (a device, six samplers, two lookups) and own a pass.
     *
     * WHAT IS IN IT, and what is deliberately not: the device; the renderer's six samplers, which a
     * declaration CHOOSES between by `sampler_hint` (a pass never names a `VkSampler` of its own, or the six
     * would become seven); the layout that occupies a shared set, asked BY SET INDEX - the same vocabulary the
     * declaration already uses; and a pass's own shader bytes, asked by name. NOT here: no instance, no
     * physical device, no allocator, no queue, no command pool, and no frame. `vulkan.core` remains the only
     * thing that creates an IMAGE, so a pass cannot take over an image family through this struct.
     */
    struct pass_context {
        /// the device a pass builds its own objects on (the owner fills this from the filtered core view)
        VkDevice device = VK_NULL_HANDLE;
        render_resource::shared::sampler_set samplers = {};
        /**
         * The layout that occupies SHARED set @p set, or `VK_NULL_HANDLE` when the owner has none there.
         *
         * A pass does not own the shared set layouts (the scene set's is `vulkan.core`'s), and it cannot build
         * its own pipeline layout without them - the pipeline layout is created from the set layouts its
         * pipeline binds, in order. Asking by SET INDEX rather than by name is what makes this the same
         * vocabulary as the declaration, which already says which set each of its bindings lives in.
         */
        VkDescriptorSetLayout (*shared_set_layout)(void* owner, uint32_t set) = nullptr;
        /**
         * The SPIR-V of one of this pass's shaders, by the name it declares; empty when the owner does not
         * have it.
         *
         * A CALLBACK rather than bytes, because a pass needs its shaders exactly once and only the ones it
         * declares - so it asks for them. The OWNER is the side that knows where shader files come from (in
         * this renderer the app loads them and hands them over), which keeps a path and a file format out of
         * this framework.
         */
        std::span<unsigned char const> (*shader)(void* owner, std::string_view name) = nullptr;
        /**
         * The SURFACE's format, which is a session-stable device fact rather than a frame's.
         *
         * WHY A PASS NEEDS IT: a pipeline that renders into the swapchain has to be created with the format that
         * image actually has, and that format is not a compile-time constant (it is whatever the surface reports;
         * `vulkan.core` finds it at startup, and `hdr_format`/`gbuffer_formats` are the constants the passes can
         * already name). The post chain's pipeline builders take it as a parameter for exactly this reason, and
         * before this field the only way to hand it over was for the runtime to build those pipelines itself -
         * which is the per-stage ownership this framework has been removing.
         *
         * The EXTENT is deliberately not here: it changes with a resize, and a pass that bakes one into an object
         * rebuilds that object in `on_swapchain_recreated` - the hook that exists for it. A format never changes
         * for a given surface, so a pass may cache this at create time.
         */
        VkFormat swap_chain_image_format = VK_FORMAT_UNDEFINED;
        /**
         * One of THIS pass's declared resources, at CREATE time: the handles its own resources are, or all-null
         * when the owner has none.
         *
         * WHY A PASS NEEDS IT, measured: a pass that owns a descriptor set built over a resource the RENDERER
         * holds - the alphaMode MASK bake's material table and bindless texture array, the compute-skinning
         * job's per-slot matrix buffers - had no way to name it, so the runtime built those sets on the pass's
         * behalf through a bespoke entry point per job. This is the channel that replaces them, and it speaks the
         * DECLARATION's vocabulary (`resource_id` + element) rather than a new one, so an owner can refuse an id
         * the pass never declared.
         *
         * THE LIFETIME RULE IS THE SAME ONE `resolved_io` FOLLOWS PER FRAME: what arrives here is a
         * session-stable handle (the runtime's material table, its texture array, a per-frame-SLOT buffer). A
         * per-swapchain-image VIEW is not stable - it is rebuilt with every generation - and those keep arriving
         * per frame through `own` / `own_per_image` / `barrier_images`, which is the channel that was added for
         * exactly that reason. A pass that cached one of those here would be storing a handle its own
         * `on_swapchain_recreated` cannot repair.
         */
        resolved_binding (*resource)(void* owner, render_resource::resource_id id, uint32_t element) = nullptr;
        /**
         * A descriptor set from the OWNER's pool, for a layout the owner handed over through
         * `shared_set_layout`.
         *
         * The pool is the core's, so a pass cannot allocate one of these itself without a pool of its own; a pass
         * that owns a PER-IMAGE family does not need this call at all (it builds the family with
         * `bindings::image_set_family`, which owns its pool and retires it correctly - see vulkan.bindings). This
         * is for the one-off set: a job that writes a single set from a shared layout and keeps it.
         */
        vk_descriptor_set (*descriptor_set)(void* owner, VkDescriptorSetLayout layout) = nullptr;
        /**
         * How many frames the owner has in flight - the number of per-frame-slot resources it will publish.
         *
         * A pass with per-slot state (the compute-skinning job's one set per slot) knows how many slots to ask
         * for only from the owner, and the alternative - looping until a slot answers with nothing - makes "the
         * owner has three slots" and "this owner forgot to publish the fourth" the same statement.
         */
        uint32_t frames_in_flight = 0;
        /// what the lookups above are called with (the renderer passes itself)
        void* owner = nullptr;
    };

    /**
     * @brief the runner's interface to the renderer: callbacks plus a context, no virtuals, no allocation
     *
     * The same injection shape `vulkan.animation`'s `backend` uses (a struct of callbacks and a context, passed
     * in rather than inherited), which is why this framework depends on NEITHER `vulkan.runtime` NOR
     * `vulkan.core`: `main.cpp`'s replacement, or a test, can supply one.
     *
     * IT IS THE RUNNER'S, NOT A PASS'S: what a pass is given is `resolved_io` at record time and
     * `pass_context` at create time. A pass has no reason to see this struct at all, which is what stops it
     * from growing into a context object that hands out whatever the newest pass wants.
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
        /// @brief build what this pass owns (its set layout, its descriptor family, its pipeline) from
        ///        @p context; once per device generation
        virtual void create(pass_context const& context) = 0;
        /// @brief the swapchain was rebuilt, so every per-image resource this pass held is stale
        virtual void on_swapchain_recreated(pass_host const& host) = 0;
        /**
         * @brief record into the frame, with the resources the declaration asked for already resolved
         *
         * NOT const, and this is a correction the first real pass forced rather than a convenience: a pass
         * that owns a descriptor family must be able to ensure it, and ensuring is what re-points the family
         * when the swapchain's views changed. The guarantee that layer actually needs is the one this keeps:
         * a pass holds no device state BETWEEN frames and is handed everything it needs to record - the
         * command buffer, its own sets, the shared ones, the pipelines, the extent and the push block. What
         * it must not do is reach for anything the run did not resolve, and that is enforced by what
         * `resolved_io` carries.
         */
        virtual void record(resolved_io const& io) = 0;
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
        /// MUTABLE pointers: a pass stores what it owns into itself through `create` and
        /// `on_swapchain_recreated`, and `record` is non-const because a pass that owns a descriptor family
        /// ensures it there (see `frame_pass::record`).
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
     *
     * The CONTEXT is the only argument, because building a pass needs nothing else: a frame loop is not
     * involved, and a caller that builds passes without ever running a frame (an editor, a test, a tool that
     * compiles pipelines) needs exactly this one struct.
     */
    [[nodiscard]] inline run_report create_stage(stage const& st, pass_context const& context) {
        run_report report;
        for (frame_pass* const p : st.passes) {
            if (p == nullptr) {
                continue;
            }
            if (!render_resource::validate(p->io())) {
                report.rejected = p->io().name;
                return report;
            }
            p->create(context);
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
