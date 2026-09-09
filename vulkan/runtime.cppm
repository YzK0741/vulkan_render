module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>

export module vulkan.runtime;
export import std;
export import vulkan.core;
export import vulkan.core.filter;
export import vulkan.runtime.scene_tree; // scene_tree owns the scene storage + GPU primitives (absorbed vulkan.model)
import utility;
export import vulkan.gui; // optional debug overlay (gui_content): exported so callers can manage panels/widgets via debug_gui()

/**
 * @file runtime.cppm
 * @defgroup vulkan_runtime Vulkan Runtime Facade
 * @brief runtime facade: a thin wrapper exposing all functionality of vulkan::core
 * @note
 *      - use operator-> to access the filtered core view (core_filter, e.g. runtime->get_device())
 *      - the inner core's lifetime is tied to the runtime
 */
namespace vulkan {
    /**
     * @ingroup vulkan_runtime
     * @brief orbit camera state, updated by the mouse callbacks registered in the runtime constructor
     */
    export struct orbit_camera {
        double last_x = 0.0;
        double last_y = 0.0;
        bool dragging = false;
        float yaw = 0.0f;
        // level view: the skybox horizon (the direction parallel to the ground plane) then sits
        // exactly at the screen center, where the primitive is framed against it
        float pitch = 0.0f;
        float distance = 2.2f;
        // point the camera looks at and orbits around (default origin; main may sink it
        // together with the scene so the camera follows the primitive)
        glm::vec3 target = glm::vec3(0.0f);
    };

    /**
     * @ingroup vulkan_runtime
     * @brief outcome of one frame step or of a whole frame (the frame phases); the caller reacts to it
     * @note shared by the split frame steps: each step returns proceed when it succeeded and the
     *       caller may continue to the next step (for the frame path that means a frame was
     *       recorded, submitted and presented). Failures are granular per stage so a caller can
     *       tell WHERE the frame broke (acquire / command-buffer recording / submit / present).
     */
    export enum class frame_status {
        proceed, // step succeeded / a frame was presented; caller continues
        skipped, // not renderable this iteration (window minimized / swapchain recreated); caller yields and retries
        closed,  // the window was closed (ESC or the native close button); caller exits the loop
        // stage-specific failures: a fatal Vulkan error at that point; caller exits the loop
        acquire_failed,         // vkAcquireNextImageKHR failed (other than out-of-date)
        begin_recording_failed, // vkBeginCommandBuffer failed
        end_recording_failed,   // vkEndCommandBuffer failed
        submit_failed,          // vkQueueSubmit failed
        present_failed,         // vkQueuePresentKHR failed (other than out-of-date / suboptimal)
    };

    /**
     * @ingroup vulkan_runtime
     * @brief true for any stage-specific failure (not proceed / skipped / closed)
     */
    export [[nodiscard]] constexpr bool is_failure(frame_status const status) noexcept {
        return status != frame_status::proceed && status != frame_status::skipped && status != frame_status::closed;
    }

    /**
     * @ingroup vulkan_runtime
     * @brief named tiers for tasks submitted to the runtime's shared task pool (run_tasks).
     *
     * Each enumerator names one frame-time parallel stage of the runtime; it maps onto the
     * underlying pool's integer priority (higher numeric values run earlier) and is ALSO that
     * stage's wait group, so distinct stages never block on each other's tasks. Values follow
     * frame causality: the stage that PRODUCES data (animation sampling) carries the higher
     * value so it runs first, the stage that CONSUMES it (command recording / rendering) the
     * lower one - "prepare the data, then render". Add an enumerator per future stage instead
     * of passing magic ints at call sites (slot new stages in by execution order; renumbering
     * is free, these values are compile-time only).
     */
    export enum class task_priority : int {
        animation = 1, // runs first: per-source animation sampling fan-out (animation::controller::update)
        recording = 0, // runs after the data is ready: pass/secondary command-buffer recording (record_main_drawcalls)
        // future frame-time stages (culling, skin upload, ...) slot in between by execution order
    };

    /**
     * @ingroup vulkan_runtime
     * @brief vulkan runtime facade class
     * @note
     *      - use operator-> to access the filtered core view (core_filter, e.g. runtime->get_device())
     *      - default construction performs the whole core initialization (window/instance/device/swap chain etc.)
     *        and registers the orbit camera mouse callbacks on the window
     */
    export class runtime {
        core vulkan_core;

        // set while the window is iconified; the restore transition recreates the swapchain
        bool was_minimized = false;

        // ---- shared scene resources (single flat descriptor set, see core::init_scene_layouts) ----
        // camera UBO: one buffer per frame slot, updated once per frame, shared by every primitive
        std::vector<vk_buffer> camera_buffers = {};
        std::vector<void*> camera_mapped = {};
        // texture registry: flat entries of the set 0 binding 1 array (raw handles); the owning
        // views / vma images live in the vectors below (vk_image RAII frees the GPU image when
        // the runtime goes away). texture_slot_cache deduplicates uploads by CONTENT (xxh3 of
        // the decoded bytes + format + dimensions): several materials sharing one glTF image
        // (same decoded pixels, different byte copies) all point at the same array slot instead
        // of uploading a copy per material - the loader hands each material its own byte copy,
        // so a pointer key would never match.
        std::vector<VkImageView> texture_array_views = {};
        std::vector<vk_image_view> owned_texture_views = {};
        std::vector<vk_image> owned_textures = {};
        uint32_t white_texture_index = 0;
        std::map<std::tuple<std::array<std::uint8_t, 16>, VkFormat, std::uint32_t, std::uint32_t, std::uint32_t>, uint32_t> texture_slot_cache = {}; // digest(128-bit), format, width, height, mip_levels
        // scene-wide IBL (bindings 2-4): prefiltered env / irradiance / BRDF LUT, uploaded once
        std::vector<vk_image_view> ibl_views = {};
        std::vector<vk_image> ibl_images = {};
        vk_sampler texture_sampler = {};
        vk_sampler env_sampler = {};
        // GPU material table (set 0 binding 5): one material_record per entry (texture indices +
        // factors + flags); primitives only push their material_index. Host-visible, written at
        // registration, read-only for the GPU.
        vk_buffer material_buffer = {};
        void* material_mapped = nullptr;
        uint32_t material_count = 0;
        // content-addressed material dedup + overflow fallback (see register_material):
        // material_slot_cache keys the full material_record bytes (texture indices + factors +
        // flags), so N primitives sharing one glTF material register ONE record instead of N
        // identical appends; when the table really fills up, later registrations degrade to the
        // reserved default material at index 0 (registered in init_scene_resources) with a
        // one-time log instead of a hard panic.
        std::map<std::array<std::uint8_t, sizeof(vulkan::material_record)>, uint32_t> material_slot_cache = {};
        bool material_overflow_logged = false;
        // per-instance transforms for instanced primitives (scene set binding 6): one mat4 per
        // instance, host-visible. The buffer is ONE shared region split into per-instanced-
        // primitive slices: make_instanced_primitive() appends its transforms at instance_cursor
        // (mat4 units), hands the slice start to the new primitive via push.instance_base, and
        // advances the cursor, so several instanced primitives coexist without overwriting each
        // other. instance_cursor resets to 0 whenever instanced primitives are cleared (they are
        // the only writers).
        vk_buffer instance_buffer = {};
        void* instance_mapped = nullptr;
        uint32_t instance_cursor = 0;
        // per-joint skin matrices (scene set binding 9): ONE buffer per frame slot, like the
        // camera UBO — each slot's scene set always points at its own buffer, so a frame being
        // rendered never shares the buffer the next frame rewrites. scene_skin_capacity mat4s
        // each, host-visible; indices 0-3 are the identity block (unskinned fallback),
        // per-skin joint blocks follow. Filled per frame by set_skin_matrices(); primitives
        // reference their block via material_push_constants::skin_base
        std::vector<vk_buffer> skin_buffers = {};
        std::vector<void*> skin_mapped = {};
        // morph data (scene set binding 10): ONE buffer per frame slot, like the skin matrices.
        // scene_morph_capacity floats each, host-visible. The caller writes per-primitive blocks
        // (morph deltas + weights) through morph_scratch() and points primitives at them via
        // material_push_constants::morph_* fields
        std::vector<vk_buffer> morph_buffers = {};
        std::vector<void*> morph_mapped = {};
        // per-slot scene descriptor sets: all pipelines share the scene layout, so every frame
        // slot gets one set from it. A set's per-slot bindings (0 camera / 7 light / 8 shadow /
        // 9 skin / 10 morph) always point at that slot's own resources and never change, so an
        // in-flight frame can never observe the next frame's descriptors (no update-after-bind
        // race).
        std::array<vk_descriptor_set, vulkan::core::MAX_FRAMES_IN_FLIGHT> scene_sets = {};
        bool scene_set_created = false;
        // the frame slot paced by the last successful pace_and_acquire();
        // per-frame host writes (set_skin_matrices / morph_scratch) target this slot's buffers
        uint32_t active_slot = 0;
        bool ibl_ready = false;
        // background pass (fullscreen triangle, no depth test): drawn first every frame
        std::optional<vk_pipeline> skybox_pipeline = std::nullopt;
        // per-stage render toggles: whether the skybox / shadow pass actually records this frame.
        // Skybox off leaves just the clear color; shadow off skips the depth pass (the shadow map
        // is cleared to fully-lit so the main pass samples "no shadow"). Both default on.
        bool skybox_enabled = true;
        bool shadow_enabled = true;

        // ---- directional shadow mapping (scene set binding 7 light UBO + binding 8 shadow map) ----
        static constexpr uint32_t shadow_map_size = 2048;
        // One shadow map per frame slot: while slot A is in flight, slot B already rewrites its
        // own map, so the two never race on the same depth image
        std::vector<vk_image> shadow_images = {}; // depth images, rendered into every frame
        std::vector<vk_image_view> shadow_image_views = {};
        vk_sampler shadow_sampler = {}; // nearest + clamp-to-edge (manual PCF in pbr.frag)
        // Light UBO (scene set binding 7): ONE host-visible buffer per frame slot, like the
        // camera UBO - each slot's scene set points at its own buffer, so the per-frame host
        // write into the paced slot's copy can never race a frame still in flight on the other
        // slot. CPU-side light_state mirrors the content: enable_shadows() fills it once,
        // set_shadow_enabled() flips its flag, and pace_and_acquire() copies it into the paced
        // slot's buffer next to the camera UBO - arbitrary-time calls (GUI callbacks included)
        // never touch mapped memory directly.
        std::vector<vk_buffer> light_buffers = {};
        std::vector<void*> light_mapped = {};
        light_ubo light_state = {};
        std::optional<vk_pipeline> shadow_pipeline = std::nullopt; // depth-only pass pipeline
        bool shadows_enabled = false;                              // true after enable_shadows() (light UBO filled + pipeline ready)
        // live-tunable depth bias of the shadow pass (dynamic state, set per frame before the
        // depth-only draw): slope-scaled bias removes acne on angled surfaces, the constant
        // factor adds a fixed push; tune from the debug gui when a model shows acne/peter-panning
        float shadow_depth_bias_constant = 0.0f;
        float shadow_depth_bias_slope = 1.5f;
        float shadow_depth_bias_clamp = 0.0f;

        // shared CPU worker pool for frame-time parallel stages (run_tasks). Sized to the
        // machine (hardware_concurrency()/4, floor 1) instead of per-consumer pools; tasks
        // are grouped by priority so each consumer waits only for its own group. Declared
        // before the scene/pipeline state so the pool outlives what tasks may touch (destructor
        // order is reverse declaration: pipelines etc. go first, the pool joins last).
        static int default_task_pool_threads() noexcept;
        utility::thread_pool task_pool = utility::thread_pool{default_task_pool_threads()};

        // Guards the pipeline registry below (pipelines / pipeline_names / default_pipeline_name):
        // parallel recording workers read it through render_environment's binder (shared locks,
        // concurrent), while make_pipeline / set_default_pipeline / resize resync write it
        // (unique lock). A reader/writer lock because reads vastly outnumber writes. Mutable so
        // const accessors (get_pipeline, the env construction inside the const record steps)
        // can take shared locks. The scene tree, per-frame vectors and camera state are NOT
        // guarded: they are only touched on the frame thread or inside run_tasks' synchronous
        // windows, so locking them would only add hot-path cost.
        mutable std::shared_mutex access_mutex;
        // string keys (not string_view): the runtime owns the pipeline names, so lookups
        // stay valid regardless of the caller's storage lifetime. std::less<> enables heterogeneous
        // lookup, so the string_view-based API (get_pipeline / ...) still works without
        // constructing a std::string per call.
        std::map<std::string, vk_pipeline, std::less<>> pipelines;
        // stable name table mirroring the pipelines map (same order as insertion): every
        // make_pipeline() appends the name, nothing removes. A DEQUE (not vector): push_back
        // never invalidates existing elements, so a pointer to it (handed to
        // render_environment::available for the recording workers' lifetime) stays valid even
        // if another thread registers a pipeline concurrently. Only make_pipeline() writes it,
        // under the unique lock; readers that took the pointer may use it lock-free afterwards.
        std::deque<std::string> pipeline_names = {};
        // name of the runtime's default pipeline: primitives with DEFAULT semantics (empty
        // pipeline_name) draw with it. Set implicitly to the FIRST created pipeline, or
        // explicitly via set_default_pipeline(); the shadow pass never consults it (its env
        // default is the shadow pipeline).
        std::string default_pipeline_name = {};
        // Scene storage: a scene tree of nodes with local transforms + children; every primitive
        // (normal_draw_primitive / instanced_draw_primitive) lives in a node's primitive leaf. The frame
        // record phase walks the tree once per frame: update_world() accumulates world matrices into each
        // leaf (primitive::set_world -> push.model), then each pipeline draws the leaves bound to
        // it (primitive::draw stays polymorphic). This replaces the old flat per-pipeline primitive list.
        //
        // NON-OWNING: the tree belongs to the caller (set_scene() binds it). The caller must keep the
        // tree alive while the runtime is in use and destroy it BEFORE the runtime (the leaves release
        // their GPU resources through this runtime's vma allocator on destruction). Declaration order in
        // the caller (scene after runtime) gives that order automatically.
        scene_tree::scene* bound_scene = nullptr; // user-owned scene the runtime renders
        // conservative radius of the bound scene around the camera target / scene center
        // (set by enable_shadows, which receives it). The camera projection far plane uses it
        // (make_orbit_camera_ubo) so zooming in never clips the scene's far side.
        float scene_radius = 100.0f;
        // optional whole-scene transform applied on top of every root before local transforms
        // (programmatic grouping / demo rotation; identity by default = no visual change)
        glm::mat4 scene_transform = glm::mat4(1.0f);
        // frustum culling of the main pass (BVH over per-leaf world AABBs vs the camera frustum);
        // enabled by default, disable for verification / debugging
        bool frustum_culling = true;
        // culling caches: the BVH is rebuilt only when the scene changed (bvh_dirty), and the
        // culled result is reused while neither the scene nor the camera moved. cull_bvh holds
        // the last built tree (world AABBs are captured at build time and stay valid as long as
        // the scene is unchanged: update_world rewrites the same matrices each frame).
        //
        // Liveness contract (why the raw primitive const* inside the BVH / cull_visible can
        // never dangle): every scene mutation that removes or adds leaves - clear_primitives,
        // make_primitive / make_instanced_primitive, import, set_scene, set_scene_transform,
        // and scene_changed() for callers editing get_scene() directly - sets bvh_dirty. The
        // next begin_recording() then (1) recollects frame_leaves from the CURRENT tree, (2)
        // destroys the old cull_bvh (its stale leaf pointers die with it) and rebuilds from
        // those fresh leaves, and (3) re-runs the cull so cull_visible also drops dead leaves.
        // Between frames the caches are never touched, so a leaf removed mid-frame is safe as
        // long as the removal went through an API that sets bvh_dirty (or the caller invoked
        // scene_changed()). Callers editing the tree behind the runtime's back must call
        // scene_changed() after every structural change or the BVH/cull_visible can outlive a
        // destroyed leaf.
        //
        // Known limitation (documented, accepted): the BVH bounds are the STATIC local AABBs
        // captured at upload/import. Skinned/morphed vertices can move far outside that box
        // (see set_skin_matrices / morph_scratch), so such a primitive may be frustum-culled
        // even while its deformed geometry is on screen. Conservative animation rigs stay
        // inside the authored bounds; culling can also be disabled (frustum_culling = false).
        std::optional<utility::bvh<primitive>> cull_bvh = std::nullopt;
        bool bvh_dirty = true;                                // scene structure/transforms changed -> rebuild
        std::pmr::vector<primitive const*> cull_visible = {}; // last culled result (main-pass set)
        // camera identity for result reuse: yaw, pitch, distance, target.xyz (7 floats)
        std::array<float, 7> camera_key = {};
        bool camera_moved = true; // camera key differs from the last cull frame
        // optional external (glTF/programmatic) camera: when active, the per-frame camera UBO
        // uses these matrices directly (the orbit camera + its mouse controls are ignored) and
        // frustum culling re-runs only when the authored view/projection actually changed
        glm::mat4 external_view = glm::mat4(1.0f);
        glm::mat4 external_proj = glm::mat4(1.0f);
        glm::vec3 external_eye = glm::vec3(0.0f);
        bool external_camera_active = false;
        bool external_camera_changed = true; // set when the override differs -> re-cull
        // one command buffer per frame slot, used and reused every frame
        std::vector<vk_command_buffer> command_buffers;
        // per-slot secondary command buffers for pass recording (stage 2/3 of parallel
        // recording):
        //   - shadow: one shadow-pass CB per frame slot (single segment; the depth-only pass
        //     shares one pipeline, so further splitting buys little - stage 2)
        //   - gui: one overlay CB per frame slot (stage 2)
        //   - transparent: one CB per frame slot for the alpha-blended leaves - recorded on the
        //     PRIMARY thread after the opaque segments (transparent leaves are usually few and
        //     order-sensitive, so they do not join the parallel fan-out), executed last before
        //     the gui overlay so blends compose over the opaque depth
        //   - main: SEGMENT secondaries per frame slot (stage 3): the main pass splits its
        //     visible leaves into up-to-SEGMENT contiguous ranges, each recorded on a pool
        //     worker and executed in order. SEGMENT is sized to the task pool (see
        //     record_main_drawcalls), so the GPU can read all secondaries while the slot's
        //     primary executes - they share the primary's lifetime (reused after the slot's
        //     timeline wait, no per-frame allocation, no pool lock).
        enum class secondary_pass : std::size_t { shadow = 0,
                                                  gui = 1,
                                                  transparent = 2,
                                                  main_seg_0 = 3, // main pass segments follow
                                                  count = 3 };    // fixed single-segment slots
        std::vector<std::array<vk_command_buffer, static_cast<std::size_t>(secondary_pass::count)>> secondary_command_buffers;
        // per-slot main-pass parallel segments (stage 3): one {command pool, secondary buffer}
        // PAIR per task-pool worker, in the same style as the vma allocator's command_cache -
        // a VkCommandPool is not thread safe, so every parallel recording thread owns its own
        // pool and the buffer allocated from it, kept together so they can never drift apart.
        // The pair's pool is registered on the core (destroyed by its cleanup AFTER this
        // runtime's RAII vk_command_buffer members free their buffers into those pools). One
        // inner vector per frame slot (the GPU reads the secondaries while the slot's primary
        // executes, so they share the primary's lifetime), index = segment.
        std::vector<std::vector<std::pair<VkCommandPool, vk_command_buffer>>> main_segments;
        // per-frame state shared by the split frame steps (the frame steps call them in order,
        // so an external caller can interleave its own work between the same steps)
        uint32_t current_image_index = 0;                      // swapchain image acquired by pace_and_acquire()
        float current_aspect = 1.0f;                           // swapchain aspect for the frame's UBO + culling
        camera_ubo current_ubo = {};                           // camera UBO snapshot written in pace_and_acquire()
        std::pmr::vector<primitive const*> frame_leaves = {};  // every scene leaf this frame (shadow + cull input)
        std::pmr::vector<primitive const*> frame_visible = {}; // opaque frustum-visible subset (main pass)
        // transparent (alphaMode BLEND) frustum-visible leaves, sorted FAR -> NEAR from the
        // camera each time the cull re-runs: drawn AFTER the opaque pass (depth-write off), so
        // the blend order is back-to-front. Rebuilt in begin_recording together with the cull.
        std::pmr::vector<primitive const*> frame_transparent = {};
        // shadow-pass subset (rebuilt each frame before the shadow recording): every leaf whose
        // shadow can reach the camera frustum = the frustum-visible leaves PLUS the leaves up to
        // shadow_caster_extent up-light of them (a caster just outside the view still throws a
        // shadow into it). cull_bvh / frame_visible only cover the CAMERA frustum, so the shadow
        // pass would otherwise re-draw every scene leaf every frame (huge on stress models like
        // NodePerformanceTest: 10000 rocks). Instanced / bound-less leaves are always included.
        std::pmr::vector<primitive const*> shadow_casters = {};
        // normalized direction toward the analytic sun (mirrors make_directional_light_ubo);
        // shadow caster culling shifts the camera frustum along this to catch up-light casters
        glm::vec3 light_direction = glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f));
        // how far up-light of the camera frustum a caster still matters (its shadow can still
        // reach the view). Scene-scale heuristic: max(1, scene_radius / 8); see enable_shadows.
        float shadow_caster_extent = 1.0f;
        // optional Dear ImGui debug overlay; inactive until enable_debug_gui() succeeds. The
        // runtime drives it inside the frame steps (new_frame before recording, record after the
        // runtime's own draw calls) so callers only manage its content via debug_gui().
        gui::gui_content debug_overlay;
        // filtered view over vulkan_core, exposed via operator-> (external code never sees the raw core)
        core_filter filtered_core;

        /**
         * @ingroup vulkan_runtime
         * @brief collect every leaf primitive under @p node (DFS pre-order) into @p out
         * @note leaves are stored as scene_tree::primitive; every leaf this runtime creates is a
         *       vulkan::primitive (the GPU primitive implements scene_tree::primitive), so the cast is safe
         */
        static void collect_leaf_primitives(scene_tree::scene_node const& node, std::pmr::vector<primitive const*>& out);
        /**
         * @ingroup vulkan_runtime
         * @brief build a normal_draw_primitive from @p info WITHOUT attaching it to the scene tree:
         *        uploads geometry buffers and registers the material (textures + material_record).
         * @param pipeline_name the pipeline the primitive draws with (must already exist)
         * @return the new primitive (caller attaches it into a scene node), or nullptr if the
         *         pipeline does not exist
         * @note make_primitive() is create_primitive() + attach-as-root-leaf; the hierarchy import
         *       (import_scene) attaches leaves to their node instead
         * @note material registration writes every scene set (binding 1), which is only valid
         *       before the first frame or while the runtime is idle - see make_primitive()'s
         *       timing note
         */
        std::unique_ptr<primitive> create_primitive(std::string_view pipeline_name, primitive_create_info const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief begin the frame's rendering on the given command buffer: clears the color
         *        attachment with a dark background and the depth attachment
         * @param command_buffer the command buffer being recorded
         * @param image_index the acquired swapchain image index (selects the attachment views)
         * @note uses vkCmdBeginRendering (dynamic rendering, Vulkan 1.3 core - the only path
         *       the engine supports; device selection requires an apiVersion >= 1.3 device)
         */
        void begin_rendering(VkCommandBuffer command_buffer, uint32_t image_index, VkRenderingFlags flags = 0) const;

        // ---- scene resource management (see the members above) ----
        void init_scene_resources();                                                          // camera UBO buffers + white fallback texture + texture sampler + material table
        void init_shadow_resources();                                                         // shadow map depth image/view/sampler + light UBO buffer
        void ensure_scene_set();                                                              // lazily create one scene set per frame slot and write all bindings
        void write_ibl_bindings() const;                                                      // (re)write bindings 2-4 on every scene set with the current IBL views / placeholders
        void write_light_and_shadow_bindings();                                               // (re)write binding 7 (light UBO) + binding 8 (shadow map) on every scene set
        void update_all_scene_sets(VkWriteDescriptorSet const* writes, uint32_t write_count); // apply one batch of writes to every slot's scene set
        uint32_t register_material(primitive_create_info const& info);                        // upload textures into the array, append a material_record, return its index

    public:
        // A non-const runtime exposes a mutable filter (e.g. runtime->get_vma()); a const runtime
        // gets a read-only filter, so mutating operations are impossible through const access.
        core_filter* operator->() noexcept {
            return &this->filtered_core;
        }
        core_filter const* operator->() const noexcept {
            return &this->filtered_core;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief orbit camera state; left-drag rotates, wheel zooms
         */
        orbit_camera camera;

        /**
         * @ingroup vulkan_runtime
         * @brief take over the per-frame camera UBO with an externally computed view/projection
         *        (e.g. a glTF camera, see the demo). While active the orbit camera mouse controls
         *        are ignored and frustum culling re-runs only when the view/projection changed.
         * @param eye camera position in world space (written into the UBO for the shaders)
         * @param view world -> camera space matrix
         * @param proj camera space -> clip space matrix (Vulkan conventions, i.e. with the Y
         *        flip already applied, same as make_orbit_camera_ubo)
         * @note call every frame while the authored camera is in use; identical matrices are
         *       deduplicated (no cull invalidation when nothing changed)
         */
        void set_external_camera(glm::vec3 const& eye, glm::mat4 const& view, glm::mat4 const& proj) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief return to the orbit camera (ignores any previously set external camera)
         */
        void clear_external_camera() noexcept;

        /** @brief whether an external (glTF/programmatic) camera currently drives the UBO */
        [[nodiscard]] bool using_external_camera() const noexcept;

        /** @brief current swapchain aspect ratio (for authored-camera projections) */
        [[nodiscard]] float aspect_ratio() const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief background clear color applied every frame (the skybox is drawn over it, so it
         *        shows only where the environment pass leaves the background uncovered)
         */
        glm::vec3 clear_color = glm::vec3(0.02f, 0.02f, 0.03f);

        /**
         * @ingroup vulkan_runtime
         * @brief run a batch of tasks on the runtime's shared worker pool and block until that
         *        priority group finished. Frame-time parallel stages (the animation controller's
         *        per-source sampling fan-out today, more later) submit their tasks here instead
         *        of owning private pools, so all CPU parallelism shares one pool sized to the
         *        machine (hardware_concurrency()/4 threads, floor 1).
         * @param tasks the batch; each task runs exactly once on a pool worker
         * @param priority the stage this batch belongs to (see task_priority); run_tasks waits
         *        only for this stage's tasks, so parallel stages can share the pool safely
         * @note synchronous: returns only after every task in the batch finished, which is what
         *       the frame phases need (the paced slot is read right after animation sampling)
         */
        void run_tasks(std::span<std::function<void()>> tasks, task_priority priority = task_priority::animation);

        /**
         * @ingroup vulkan_runtime
         * @brief worker count of the shared task pool (what run_tasks() fans out over); callers
         *        that need to slice their work across workers (e.g. the animation controller's
         *        per-source sampling) size their slices to this
         */
        [[nodiscard]] int task_pool_threads() const noexcept {
            return this->task_pool.thread_count();
        }

        /**
         * @ingroup vulkan_runtime
         * @brief construct the runtime: performs the full core initialization (window / instance /
         *        device / swapchain / resources) from @p options (window size, vsync, MSAA), and
         *        registers the orbit camera mouse callbacks on the window
         */
        explicit runtime(core_create_info const& options);

        /**
         * @ingroup vulkan_runtime
         * @brief construct the runtime with default core options (1080x960 window, auto MSAA,
         *        mailbox present mode)
         */
        runtime();

        /**
         * @ingroup vulkan_runtime
         * @brief destroy cached pipelines before the inner core (and thus the VkDevice) is destroyed
         * @note explicit destructor: member destruction order is reverse declaration order, which
         *      currently already destroys pipelines before vulkan_core; making it explicit keeps
         *      that guarantee even if members are reordered later
         */
        ~runtime();

        /**
         * @ingroup vulkan_runtime
         * @brief initialize the Dear ImGui debug overlay on top of this runtime's window
         * @return true when the overlay is active afterwards (initialized, or already active)
         * @note the overlay is drawn inside the runtime frame phases: its
         *       per-frame new_frame/record calls are driven by the runtime once enabled. Call
         *       after the runtime is fully set up (window/device ready). Safe to call again to
         *       re-enable after shutdown; no-op when already active.
         */
        bool enable_debug_gui();

        /**
         * @ingroup vulkan_runtime
         * @brief true while the Dear ImGui debug overlay is active
         */
        [[nodiscard]] bool debug_gui_active() const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief access the debug overlay to manage its content from outside (register panels,
         *        push widgets, show/hide windows)
         * @return the runtime's gui_content (non-const: adding/removing panels mutates it)
         * @note panels added here are drawn every rendered frame by the runtime; add them after
         *       enable_debug_gui() (or any time — they are only drawn while the overlay is active)
         */
        [[nodiscard]] gui::gui_content& debug_gui() noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief drive one whole application frame in a single call by running the frame phases
         *        directly (see below): poll/skip/close, recreate-if-minimized, pace + acquire,
         *        record (world accumulation / culling / shadow / main pass / overlay) and
         *        submit + present, with no opportunity to interleave per-frame host writes.
         * @return frame_status: proceed when a frame was presented; skipped when not renderable
         *         (minimized or swapchain recreated - caller yields and calls again); closed on
         *         window close; one of the stage-specific *_failed values on a fatal Vulkan
         *         error (is_failure() tests for any of them)
         * @note callers that must write per-frame data (animation poses, skin matrices, morph
         *       weights) between pacing and recording call the granular phases themselves in the
         *       same order: poll_events -> recreate_if_minimized -> pace_and_acquire, then the
         *       writes (they may only touch per-slot buffers after pace_and_acquire has waited
         *       the slot's timeline), then begin_recording -> record_main_drawcalls ->
         *       end_recording -> submit_and_present.
         */
        frame_status render_frame();

        /**
         * @ingroup vulkan_runtime
         * @brief the frame slot paced by the last successful pace_and_acquire(); per-frame host
         *        writes (set_skin_matrices / morph_scratch) go into this slot's per-slot buffers
         */
        [[nodiscard]] uint32_t active_frame_slot() const noexcept {
            return this->active_slot;
        }

        // ---- frame phases: one frame is these calls in order. render_frame() runs them all
        //      back to back; callers with per-frame host writes run them at fine granularity
        //      (writes go between pace_and_acquire() and begin_recording()) ----
        /** @brief poll window events; returns closed on ESC/native close, skipped while minimized */
        frame_status poll_events();
        /** @brief recreate the swapchain if the window was minimized (extent 0) since the last frame */
        void recreate_if_minimized();
        /** @brief wait the frame slot's timeline, acquire the next image, write the camera UBO,
         *         and remember the paced slot in active_frame_slot(); skipped when the swapchain
         *         was recreated, acquire_failed on a fatal acquire error */
        frame_status pace_and_acquire();
        /** @brief begin the slot's command buffer and run the CPU scene prep (world-matrix
         *         accumulation + frustum culling); begin_recording_failed when vkBeginCommandBuffer fails */
        frame_status begin_recording();
        /** @brief record the shadow pass, the attachment transitions and the main scene pass */
        void record_main_drawcalls();

        /**
         * @ingroup vulkan_runtime
         * @brief record the depth-only shadow-pass drawing content into @p command_buffer:
         *        bind the shared scene set + shadow pipeline, set the live depth bias, draw
         *        every scene leaf. The caller frames it (already inside the shadow rendering
         *        instance, depth-only).
         * @note extracted from record_main_drawcalls() so the same content can be recorded
         *       inline (stage 1) or into a per-slot secondary command buffer (stage 2,
         *       parallel recording) - only bind/push/draw commands, no barriers / begin-end.
         */
        void record_shadow_content(VkCommandBuffer command_buffer) const;

        /**
         * @ingroup vulkan_runtime
         * @brief record the main-pass scene content into @p command_buffer: bind the shared
         *        scene set, draw the skybox background (when enabled) then every pipeline's
         *        visible leaves. The caller frames it (already inside the main rendering
         *        instance with color+depth attachments).
         * @note extracted from record_main_drawcalls() so the same content can be recorded
         *       inline (stage 1) or into a per-slot secondary command buffer (stage 2,
         *       parallel recording) - only bind/push/draw commands, no barriers / begin-end.
         */
        void record_main_content(VkCommandBuffer command_buffer) const;

        /**
         * @ingroup vulkan_runtime
         * @brief record one contiguous slice of the main-pass leaves into @p command_buffer:
         *        bind the shared scene set, then draw the leaves of @p leaves (a sub-range of
         *        frame_visible). When @p draw_skybox the skybox background is drawn first so
         *        the background stays ordered before the scene (segment 0 only); later
         *        segments are pure scene.
         * @note stage 3 of parallel recording: each task-pool worker records one segment into
         *       its own secondary command buffer (see sub_render_task), the primary executes
         *       them in order. Only bind/push/draw commands - caller owns barriers + the
         *       rendering instance.
         */
        void record_main_segment(VkCommandBuffer command_buffer, std::span<primitive const* const> leaves, bool draw_skybox) const;

        /**
         * @ingroup vulkan_runtime
         * @brief one recording job of the parallel main pass (stage 3): records @p leaves (a
         *        contiguous slice of the frame's visible leaves) into @p command_buffer, a
         *        per-slot SECONDARY command buffer. operator() begins the secondary (inheriting
         *        the main instance's color+depth attachments via dynamic rendering 1.3
         *        inheritance info), records the slice and ends it, so a batch of these can be
         *        posted straight to the shared task pool and the recording group waited on.
         * @note value type (span + handle + formats; no owning pointers), safe to copy into
         *       std::function for the pool; the begin-info is assembled fresh inside operator()
         *       so copies never share dangling pNext chains.
         */
        struct sub_render_task {
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            std::span<primitive const* const> leaves = {};
            bool draw_skybox = false;                    // segment 0 draws the skybox before its leaves
            VkFormat color_format = VK_FORMAT_UNDEFINED; // main color attachment format
            VkFormat depth_format = VK_FORMAT_UNDEFINED; // main depth attachment format
            VkSampleCountFlagBits rasterization_samples = VK_SAMPLE_COUNT_1_BIT;
            runtime const* owner = nullptr; // recording context (scene set / pipeline caches)
            // set to true by operator() when the secondary was actually recorded (begin + end
            // succeeded). Points into a per-frame array owned by the caller of the task batch;
            // the caller waits the recording group before reading it, so no extra sync is
            // needed. The primary must NOT execute a segment whose begin failed.
            std::atomic<bool>* recorded = nullptr;

            void operator()() const; // defined in runtime.cpp (module-private)
        };

        /** @brief the command buffer currently being recorded (between begin_recording() and
         *         end_recording()); internal use for the runtime's own recording */
        [[nodiscard]] VkCommandBuffer active_command_buffer() const noexcept;
        /** @brief end the rendering instance / render pass and finish recording;
         *         end_recording_failed when vkEndCommandBuffer fails */
        frame_status end_recording();
        /** @brief submit the recorded command buffer and present the image (recreating the
         *         swapchain when presentation reports out of date); submit_failed /
         *         present_failed on fatal errors */
        frame_status submit_and_present();

        /**
         * @ingroup vulkan_runtime
         * @brief create a named pipeline from raw SPIR-V and cache it in the runtime. The first
         *        pipeline created becomes the runtime's DEFAULT pipeline (implicitly); primitives
         *        with default semantics draw with it. Every pipeline shares the scene set layout,
         *        so any number of them can coexist in one scene (leaves choose by name).
         * @param pipeline_name the pipeline's name (used by primitives to request it, and by
         *        render_environment to bind it); must be unique
         * @param vertex_shader_code raw SPIR-V binary of the vertex shader
         * @param fragment_shader_code raw SPIR-V binary of the fragment shader
         * @return success, or an error message on failure
         * @note thread-safe (registry guarded), but call OUTSIDE the frame loop: recording
         *       workers read the registry lock-free during a frame (see record_main_segment's
         *       concurrency note), so mid-frame registration would race them. Register at setup
         *       or between frames (while idle).
         */
        std::expected<void, std::string> make_pipeline(
            std::string_view pipeline_name,
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief make @p pipeline_name the runtime's default pipeline (the one default-semantics
         *        primitives draw with; see make_pipeline for the implicit first-pipeline default)
         * @param pipeline_name a pipeline previously created via make_pipeline()
         */
        void set_default_pipeline(std::string_view pipeline_name);

        /**
         * @ingroup vulkan_runtime
         * @brief create the skybox background pipeline: a fullscreen triangle (drawn with
         *        vkCmdDraw(3), no vertex/index buffers) that samples the environment cubemap
         * @param vertex_shader_code raw SPIR-V binary of the skybox vertex shader
         * @param fragment_shader_code raw SPIR-V binary of the skybox fragment shader
         * @return success, or an error message on failure
         * @note drawn first in every frame with depth test/write disabled, so models render over it;
         *       uses the shared scene set (camera UBO binding 0, env cubemap binding 2)
         */
        std::expected<void, std::string> make_skybox_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief create the directional shadow pipeline: renders scene geometry depth-only into
         *        the shadow map (no color attachment), from the light's view
         * @param vertex_shader_code raw SPIR-V binary of the shadow vertex shader
         * @param fragment_shader_code raw SPIR-V binary of the shadow fragment shader
         * @return success, or an error message on failure
         * @note depth-only rendering (no color attachment) requires dynamic rendering, which
         *       is core Vulkan 1.3 - the only path the engine supports
         */
        std::expected<void, std::string> make_shadow_pipeline(
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code);

        /**
         * @ingroup vulkan_runtime
         * @brief enable directional shadow mapping: fills the light UBO with an orthographic
         *        view-proj framing the given scene bounds (plus the light direction, matching
         *        the sky sun). Must be called after the models exist (the shadow pass draws them).
         * @param scene_center world-space center of the shadow frustum (usually the scene bounds
         *        center after the scene offset is applied, i.e. where the models actually sit)
         * @param scene_radius conservative radius covering all shadow casters
         * @note requires make_shadow_pipeline() to have succeeded; no-op otherwise
         */
        void enable_shadows(glm::vec3 const& scene_center, float scene_radius);

        /**
         * @ingroup vulkan_runtime
         * @brief set an extra whole-scene transform applied on top of every scene-tree root
         * @param transform world matrix placed before the roots' local transforms
         * @note identity (the default) leaves rendering untouched; useful for programmatic
         *       grouping / demo rotation of the whole imported scene
         */
        void set_scene_transform(glm::mat4 const& transform);

        /**
         * @ingroup vulkan_runtime
         * @brief enable or disable main-pass frustum culling (BVH vs camera frustum)
         * @param enabled true (default) culls leaves outside the view frustum before drawing
         */
        void set_frustum_culling(bool enabled) noexcept {
            this->frustum_culling = enabled;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief enable or disable drawing the skybox background pass each frame
         * @param enabled true (default) draws the environment skybox; false leaves the clear color
         * @note cheap toggle: only affects command recording, no resource rebuild
         */
        void set_skybox_enabled(bool enabled) noexcept {
            this->skybox_enabled = enabled;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief enable or disable recording the directional shadow pass each frame
         * @param enabled true (default) renders the shadow map; false skips the depth pass and
         *        clears the shadow map to fully-lit so the main pass shows no shadows
         * @note requires enable_shadows() to have succeeded; turning it off clears the shadow maps
         *       (a one-shot GPU command), re-enabling restores per-frame rendering
         */
        void set_shadow_enabled(bool enabled);

        /**
         * @ingroup vulkan_runtime
         * @brief set the live depth bias of the directional shadow pass (applied every frame via
         *        vkCmdSetDepthBias before the depth-only draw)
         * @param constant_factor fixed depth bias added to every fragment's depth
         * @param slope_factor slope-scaled bias (depth units per depth-unit of surface slope) -
         *        the main acne control for angled surfaces
         * @param clamp maximum depth bias magnitude (0 = no clamp)
         * @note cheap: only changes the value recorded into the command buffer, no pipeline or
         *       resource rebuild; useful to chase shadow acne / peter-panning per model
         */
        void set_shadow_depth_bias(float constant_factor, float slope_factor, float clamp) noexcept {
            this->shadow_depth_bias_constant = constant_factor;
            this->shadow_depth_bias_slope = slope_factor;
            this->shadow_depth_bias_clamp = clamp;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief mark the scene tree as changed (structure or per-node local transforms edited
         *        through get_scene(), e.g. programmatic animation): the culling BVH is rebuilt on
         *        the next frame. Internal scene mutations (import / make / clear /
         *        set_scene_transform) invalidate automatically.
         */
        void scene_changed() noexcept {
            this->bvh_dirty = true;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief print the scene tree (names + local-transform marker + leaf primitive pipeline)
         *        to the log, one indented line per node, plus a shape summary
         * @note diagnostic helper: shows whether an import rebuilt the real hierarchy (gltf
         *       node names and nesting) or a flat list of root leaves (pipeline names)
         */
        void log_scene_tree() const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief bind the scene tree this runtime renders. The tree is owned by the caller
         *        (never by the runtime): it must stay alive while the runtime is in use and be
         *        destroyed before the runtime goes away, because the leaves' GPU buffers are
         *        released through the runtime's vma allocator when the tree is destroyed.
         * @param scene the caller-owned scene tree to render
         * @note bind before creating/importing primitives and before the first frame; the
         *       culling BVH is invalidated so the next frame rebuilds it
         */
        void set_scene(scene_tree::scene& scene) noexcept {
            this->bound_scene = &scene;
            this->bvh_dirty = true; // a new tree's world AABBs must be (re)collected
        }

        /**
         * @ingroup vulkan_runtime
         * @brief access the scene tree (roots + children + per-node local transforms) for
         *        programmatic whole-group / subtree transforms
         * @note the tree structure is fixed after import (no reallocation of scene or the
         *       children vectors while nodes are only edited in place), so pointers/references
         *       into the tree stay valid until the next make_primitive / import / clear call
         */
        [[nodiscard]] scene_tree::scene& get_scene() noexcept {
            if (this->bound_scene == nullptr) {
                utility::panic("runtime::get_scene() called before set_scene() bound a scene");
            }
            return *this->bound_scene;
        }
        [[nodiscard]] scene_tree::scene const& get_scene() const noexcept {
            if (this->bound_scene == nullptr) {
                utility::panic("runtime::get_scene() called before set_scene() bound a scene");
            }
            return *this->bound_scene;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief get a cached pipeline by its name
         * @param pipeline_name the name passed to make_pipeline()
         * @return pointer to the cached pipeline, or nullptr if no pipeline with that name exists
         */
        [[nodiscard]] vk_pipeline const* get_pipeline(std::string_view pipeline_name) const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief upload the scene-wide IBL resources (prefiltered env / irradiance / BRDF LUT)
         *        into the shared scene set; call it before creating models that use IBL
         * @param info precomputed split-sum IBL bytes (see vulkan::generate_* helpers)
         * @note the images are uploaded once and shared by every primitive (they used to be
         *       duplicated per primitive)
         * @note call before the first frame, or only while the runtime is idle (no frame in
         *       flight): this rewrites bindings 2-4 on every scene set, and the scene sets are
         *       no longer update-after-bind - updating a set an in-flight frame may read is a
         *       spec violation. Mid-loop IBL swaps need wait_idle() first.
         */
        void set_ibl(ibl_input const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief upload the scene-wide skin matrices (scene set binding 9) into the frame slot
         *        paced by the last pace_and_acquire(): the caller fills the buffer layout
         *        [identity block (4 mat4s) | per-skin joint blocks] and calls this once per frame
         *        AFTER pace_and_acquire(); primitives reference their block start via
         *        material_push_constants::skin_base (0 = the identity block: unskinned draws)
         * @param matrices at most scene_skin_capacity mat4s; anything beyond the capacity is dropped
         * @note host-visible copy, no Vulkan objects involved; the per-slot buffers guarantee an
         *       in-flight frame never shares the buffer being rewritten
         */
        void set_skin_matrices(std::span<glm::mat4 const> matrices);

        /**
         * @ingroup vulkan_runtime
         * @brief like set_skin_matrices() but into an explicit slot buffer (used for setup-time
         *        uploads that must be visible to every slot, e.g. the identity block before the
         *        render loop starts)
         */
        void set_skin_matrices(std::span<glm::mat4 const> matrices, uint32_t slot);

        /**
         * @ingroup vulkan_runtime
         * @brief host-visible scratch memory of the ACTIVE frame slot's morph buffer (scene set
         *        binding 10, scene_morph_capacity floats each). The caller lays out per-primitive
         *        morph blocks (per vertex per target pos-delta/nrm-delta floats, then the
         *        per-target weights) and points primitives at their block through
         *        material_push_constants::morph_base / morph_targets / morph_vertices (see those
         *        fields for the layout convention).
         * @return the mapped base of the slot paced by the last pace_and_acquire(), or nullptr when
         *         the morph buffer is unavailable
         */
        [[nodiscard]] void* morph_scratch() noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief like morph_scratch() but for an explicit slot buffer (used for setup-time bakes
         *        that must be duplicated into every slot's buffer before the render loop starts)
         */
        [[nodiscard]] void* morph_scratch(uint32_t slot) noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief create a primitive and attach it to the scene tree as a new leaf (root node).
         * @param pipeline_name the pipeline the primitive draws with (must already exist)
         * @param info geometry and material textures
         * @return pointer to the created primitive (owned by the scene tree), or nullptr if the
         *         pipeline does not exist
         * @note the leaf's local transform is @p info.model_matrix and its world is identity
         *       (the frame record phase runs update_world before drawing, so primitive::set_world writes
         *       the same matrix into push.model as before)
         * @note call before the first frame, or only while the runtime is idle (no frame in
         *       flight): registering a material appends binding-1 texture entries to every scene
         *       set, and the scene sets are no longer update-after-bind - updating a set an
         *       in-flight frame may read is a spec violation. Mid-loop creation needs wait_idle()
         *       first.
         */
        primitive* make_primitive(std::string_view pipeline_name, primitive_create_info const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief collect every leaf primitive of the given pipeline (DFS over the scene tree)
         * @param pipeline_name the pipeline name passed to make_primitive()
         * @return models whose leaf node name matches @p pipeline_name, in scene-tree order
         */
        [[nodiscard]] std::vector<primitive const*> get_primitives(std::string_view pipeline_name) const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief append an instanced_draw_primitive: draws @p source's geometry once per transform
         *        in ONE draw call per frame (per-instance matrices in scene set binding 6)
         * @param source any primitive of this runtime (its geometry is drawn transforms.size() times;
         *        it must stay in the runtime's primitive list while the instanced primitive is drawn)
         * @param transforms one world matrix per instance (fully places the source geometry)
         * @return pointer to the appended instanced primitive, or nullptr if nothing was appended
         */
        primitive* make_instanced_primitive(primitive const& source, std::span<glm::mat4 const> transforms);

        /**
         * @ingroup vulkan_runtime
         * @brief append a static_draw_primitive: OWNS one merged vertex/index buffer and draws
         *        a chunk table over it — one buffer bind, then one offset draw per chunk with
         *        each chunk's own material. The primitive-level form of a static scene: N
         *        static sub-meshes cost 1 bind + N draws instead of N binds + N draws.
         * @param info merged geometry + chunk table (the packer's output; empty chunks draw the
         *        whole merged range once, degenerating to a plain normal draw)
         * @return pointer to the appended static primitive, or nullptr if nothing was appended
         * @note the whole batch shares one world (push.model from update_world); per-chunk
         *       placement needs separate batches or baked chunk models
         */
        primitive* make_static_draw(static_draw_create_info const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief batch-import a scene by traversing the retained node hierarchy (structural
         *        node stream) and its drawables (geometry stream) together.
         *        @p nfirst must model vulkan::scene_node_iterator: DFS pre-order over every
         *        scene node INCLUDING transform-only nodes, exposing get_name() /
         *        get_local_transform() / get_depth() / get_drawable_count(). @p dfirst must
         *        model vulkan::scene_drawable_iterator (++ plus geometry/material getters);
         *        the loader keeps both streams over the same pool in the same order, so each
         *        node's get_drawable_count() drawables are the next entries of the drawable
         *        stream. The runtime drives the traversal: it rebuilds the node tree into
         *        scene_tree::scene (one scene_node per loader node, named, with its local
         *        transform; a node's drawable becomes a primitive leaf attached to that node —
         *        extra primitives of one node become identity-local child leaves), uploads
         *        buffers and registers materials. No glTF (or any scene format) knowledge
         *        lives in the runtime.
         * @param nfirst,nlast iterator pair over the scene's node hierarchy
         * @param dfirst,dlast iterator pair over the scene's drawables
         * @param offset translation applied to every scene ROOT node's local transform
         *        (e.g. -scene_center + sink); children inherit it through update_world
         * @return counts of imported primitives and materials
         * @note call before the first frame, or only while the runtime is idle (no frame in
         *       flight): importing registers materials, which appends binding-1 texture entries
         *       to every scene set, and the scene sets are no longer update-after-bind -
         *       updating a set an in-flight frame may read is a spec violation. Mid-loop imports
         *       need wait_idle() first.
         */
        template <class NI, class DI>
            requires scene_node_iterator<NI> && scene_drawable_iterator<DI>
        scene_import_result import_scene(NI nfirst, NI nlast, DI dfirst, DI dlast, glm::vec3 const& offset) {
            if (this->bound_scene == nullptr) {
                utility::panic("runtime::import_scene() called before set_scene() bound a scene");
            }
            scene_import_result result = {};
            uint32_t const materials_before = this->material_count;
            // converts a pure image_source (e.g. the glTF loader's image_view) into the
            // internal texture_input with the slot's upload format; invalid images -> white
            auto const to_texture = [](auto const& image, VkFormat const format) {
                texture_input out = {};
                if (image.valid) {
                    out.data = image.data;
                    out.width = image.width;
                    out.height = image.height;
                    out.mip_levels = image.mip_levels;
                    out.format = format;
                    out.valid = true;
                }
                return out;
            };
            // per-node drawable -> primitive_create_info (reads the next drawable of the stream)
            auto const fill_info = [&](DI& drawable, primitive_create_info& info) {
                auto const vertex = drawable.get_vertex();
                auto const index = drawable.get_index();
                info.vertex_data = vertex.data;
                info.vertex_stride = vertex.stride;
                info.vertex_count = vertex.count;
                info.index_data = index.data;
                info.index_type = index.width == 4 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
                info.index_count = index.count;
                info.albedo = to_texture(drawable.get_albedo(), VK_FORMAT_R8G8B8A8_SRGB);
                info.metallic_roughness = to_texture(drawable.get_metallic_roughness(), VK_FORMAT_R8G8B8A8_UNORM);
                info.normal = to_texture(drawable.get_normal(), VK_FORMAT_R8G8B8A8_UNORM);
                info.occlusion = to_texture(drawable.get_occlusion(), VK_FORMAT_R8G8B8A8_UNORM);
                info.emissive = to_texture(drawable.get_emissive(), VK_FORMAT_R8G8B8A8_SRGB); // glTF emissive textures are sRGB
                auto const factors = drawable.get_factors();
                info.factors.base_color_factor = factors.base_color_factor;
                info.factors.emissive_factor = factors.emissive_factor;
                info.factors.metallic_factor = factors.metallic_factor;
                info.factors.roughness_factor = factors.roughness_factor;
                info.factors.normal_scale = factors.normal_scale;
                info.factors.occlusion_strength = factors.occlusion_strength;
                info.factors.alpha_cutoff = factors.alpha_cutoff;
                info.factors.alpha_mask = factors.alpha_mask;
                info.factors.alpha_blend = factors.alpha_blend;
                info.double_sided = drawable.get_double_sided();
            };
            // attach one leaf primitive to @p node (geometry from the next drawable of the stream);
            // returns the created primitive or nullptr if the pipeline is missing
            auto const attach_leaf = [&](scene_tree::scene_node& node, DI& drawable) -> primitive* {
                if (!(drawable != dlast)) {
                    utility::panic(std::source_location::current(), "drawable stream ended before the node tree did");
                }
                primitive_create_info info = {};
                fill_info(drawable, info);
                ++drawable;
                ++result.primitive_count;
                std::unique_ptr<primitive> created = this->create_primitive("pbr", info);
                if (created == nullptr) {
                    return nullptr;
                }
                if (node.primitive_leaf == nullptr) {
                    node.primitive_leaf = std::move(created);
                } else {
                    // a glTF node can carry several primitives; scene_node has one leaf slot, so
                    // extra primitives become identity-local child leaves (world unchanged)
                    scene_tree::scene_node extra;
                    extra.name = node.name + "/prim";
                    extra.primitive_leaf = std::move(created);
                    node.children.push_back(std::move(extra));
                }
                return static_cast<primitive*>(node.primitive_leaf ? node.primitive_leaf.get() : node.children.back().primitive_leaf.get());
            };

            // DFS over the loader's node stream, rebuilding parent/child edges with an explicit
            // stack of ancestors: ancestors[d] holds the scene_node at depth d on the path to
            // the current node. The loader emits nodes in DFS pre-order, so when a node arrives
            // at depth d its parent is the ancestor at depth d-1 (pop everything deeper first).
            std::vector<scene_tree::scene_node*> ancestors = {}; // ancestors[d] = node at depth d
            for (; nfirst != nlast; ++nfirst) {
                std::size_t const depth = nfirst.get_depth();
                // pop ancestors deeper than the arriving node's parent level (their subtrees are done)
                while (ancestors.size() > depth) {
                    ancestors.pop_back();
                }
                scene_tree::scene_node node;
                node.name = std::string(nfirst.get_name());
                node.source_index = nfirst.get_source_index(); // asset node index (e.g. glTF): animation targets map onto the tree through it
                node.local = nfirst.get_local_transform();
                if (depth == 0) {
                    node.local = glm::translate(glm::mat4(1.0f), offset) * node.local; // scene root gets the offset
                }
                // attach under the parent (depth-1) or as a new scene root
                if (depth == 0) {
                    this->bound_scene->roots.push_back(std::move(node));
                    ancestors.assign(1, &this->bound_scene->roots.back());
                } else {
                    if (ancestors.size() != depth) {
                        utility::panic(std::source_location::current(), "node tree stream: broken ancestor stack");
                    }
                    scene_tree::scene_node* const parent = ancestors[depth - 1];
                    parent->children.push_back(std::move(node));
                    ancestors.resize(depth + 1);
                    ancestors[depth] = &parent->children.back();
                }
                scene_tree::scene_node* const current = ancestors[depth];
                // consume this node's drawables (the stream is aligned node-for-node)
                std::size_t const drawable_count = nfirst.get_drawable_count();
                for (std::size_t i = 0; i < drawable_count; ++i) {
                    if (attach_leaf(*current, dfirst) == nullptr) {
                        utility::panic(std::source_location::current(), "failed to import drawable (pipeline 'pbr' missing)");
                    }
                }
            }
            this->bvh_dirty = true; // new leaves attached -> culling BVH must be rebuilt
            result.material_count = this->material_count - materials_before;
            return result;
        }

        /**
         * @ingroup vulkan_runtime
         * @brief destroy and remove all models of a pipeline, no-op if the pipeline has none
         * @param pipeline_name the pipeline name passed to make_primitive()
         */
        void clear_primitives(std::string_view pipeline_name);
    };
} // namespace vulkan