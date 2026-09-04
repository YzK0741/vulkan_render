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
     * @brief outcome of one frame step or of the whole render_frame(); the caller reacts to it
     * @note shared by the split frame steps: each step returns proceed when it succeeded and the
     *       caller may continue to the next step (for render_frame() that means a frame was
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
        std::vector<uint64_t> camera_buffer_handles = {};
        std::vector<void*> camera_mapped = {};
        // texture registry: flat entries of the set 0 binding 1 array (raw handles); the owning
        // views / vma handles live in the vectors below. texture_slot_cache deduplicates uploads:
        // several materials sharing one glTF texture (same decoded bytes) all point at the same
        // array slot instead of uploading a copy per material.
        std::vector<VkImageView> texture_array_views = {};
        std::vector<vk_image_view> owned_texture_views = {};
        std::vector<uint64_t> owned_texture_handles = {};
        uint32_t white_texture_index = 0;
        std::map<std::tuple<unsigned char const*, std::size_t, VkFormat>, uint32_t> texture_slot_cache = {};
        // scene-wide IBL (bindings 2-4): prefiltered env / irradiance / BRDF LUT, uploaded once
        std::vector<vk_image_view> ibl_views = {};
        std::vector<uint64_t> ibl_handles = {};
        vk_sampler texture_sampler = {};
        vk_sampler env_sampler = {};
        // GPU material table (set 0 binding 5): one material_record per entry (texture indices +
        // factors + flags); primitives only push their material_index. Host-visible, written at
        // registration, read-only for the GPU.
        uint64_t material_buffer_handle = 0;
        void* material_mapped = nullptr;
        uint32_t material_count = 0;
        // per-instance transforms for instanced primitives (scene set binding 6): one mat4 per
        // instance, host-visible; filled by make_instanced_primitive()
        uint64_t instance_buffer_handle = 0;
        void* instance_mapped = nullptr;
        // the single scene descriptor set: all pipelines share the layout, so one set covers them all
        vk_descriptor_set scene_set = {};
        bool scene_set_created = false;
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
        std::vector<uint64_t> shadow_image_handles = {}; // depth images, rendered into every frame
        std::vector<vk_image_view> shadow_image_views = {};
        vk_sampler shadow_sampler = {};   // nearest + clamp-to-edge (manual PCF in pbr.frag)
        uint64_t light_buffer_handle = 0; // host-visible light UBO (static content)
        void* light_mapped = nullptr;
        std::optional<vk_pipeline> shadow_pipeline = std::nullopt; // depth-only pass pipeline
        bool shadows_enabled = false;                              // true after enable_shadows() (light UBO filled + pipeline ready)

        std::mutex access_mutex;
        // string keys (not string_view): the runtime owns the pipeline names, so lookups
        // stay valid regardless of the caller's storage lifetime. std::less<> enables heterogeneous
        // lookup, so the string_view-based API (get_pipeline / ...) still works without
        // constructing a std::string per call.
        std::map<std::string, vk_pipeline, std::less<>> pipelines;
        // Scene storage: a scene tree of nodes with local transforms + children; every primitive
        // (normal_draw_primitive / instanced_draw_primitive) lives in a node's primitive leaf. render_frame
        // walks the tree once per frame: update_world() accumulates world matrices into each
        // leaf (primitive::set_world -> push.model), then each pipeline draws the leaves bound to
        // it (primitive::draw stays polymorphic). This replaces the old flat per-pipeline primitive list.
        scene_tree::scene scene = {}; // single scene; roots own every primitive leaf
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
        std::optional<utility::bvh<primitive>> cull_bvh = std::nullopt;
        bool bvh_dirty = true;                           // scene structure/transforms changed -> rebuild
        std::vector<primitive const*> cull_visible = {}; // last culled result (main-pass set)
        // camera identity for result reuse: yaw, pitch, distance, target.xyz (7 floats)
        std::array<float, 7> camera_key = {};
        bool camera_moved = true; // camera key differs from the last cull frame
        // one command buffer per frame slot, used and reused by render_frame()
        std::vector<vk_command_buffer> command_buffers;
        // per-frame state shared by the split frame steps (render_frame() calls them in order,
        // so an external caller can interleave its own work between the same steps)
        uint32_t current_image_index = 0;                 // swapchain image acquired by set_up_frame_environment()
        float current_aspect = 1.0f;                      // swapchain aspect for the frame's UBO + culling
        camera_ubo current_ubo = {};                      // camera UBO snapshot written in set_up_frame_environment()
        std::vector<primitive const*> frame_leaves = {};  // every scene leaf this frame (shadow + cull input)
        std::vector<primitive const*> frame_visible = {}; // frustum-visible subset (main pass)
        std::size_t frame_culled_count = 0;               // leaves culled this frame (for the log)
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
        void collect_leaf_primitives(scene_tree::scene_node const& node, std::vector<primitive const*>& out) const;
        /**
         * @ingroup vulkan_runtime
         * @brief destroy every leaf primitive under @p node (recursively) with @p vma
         */
        void destroy_leaf_primitives(scene_tree::scene_node& node, vma_allocator& vma);
        /**
         * @ingroup vulkan_runtime
         * @brief build a normal_draw_primitive from @p info WITHOUT attaching it to the scene tree:
         *        uploads geometry buffers and registers the material (textures + material_record).
         * @param pipeline_name the pipeline the primitive draws with (must already exist)
         * @return the new primitive (caller attaches it into a scene node), or nullptr if the
         *         pipeline does not exist
         * @note make_primitive() is create_primitive() + attach-as-root-leaf; the hierarchy import
         *       (import_scene) attaches leaves to their node instead
         */
        std::unique_ptr<primitive> create_primitive(std::string_view pipeline_name, primitive_create_info const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief begin the frame's rendering on the given command buffer: clears the color
         *        attachment with a dark background and the depth attachment
         * @param command_buffer the command buffer being recorded
         * @param image_index the acquired swapchain image index (selects the attachment views)
         * @note uses vkCmdBeginRendering (dynamic rendering) when the device supports it,
         *       otherwise falls back to the classic render pass + framebuffer path
         */
        void begin_rendering(VkCommandBuffer command_buffer, uint32_t image_index) const;

        // ---- scene resource management (see the members above) ----
        void init_scene_resources();                                   // camera UBO buffers + white fallback texture + texture sampler + material table
        void init_shadow_resources();                                  // shadow map depth image/view/sampler + light UBO buffer
        void ensure_scene_set();                                       // lazily create the scene set and write camera + IBL + material bindings
        void write_ibl_bindings() const;                               // (re)write bindings 2-4 with the current IBL views / placeholders
        void write_light_and_shadow_bindings();                        // (re)write binding 7 (light UBO) + binding 8 (shadow map)
        uint32_t register_material(primitive_create_info const& info); // upload textures into the array, append a material_record, return its index
        void clear_shadow_maps_to_lit();                               // one-shot GPU clear of every frame slot's shadow map to depth 1.0

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
         * @brief background clear color applied every frame (the skybox is drawn over it, so it
         *        shows only where the environment pass leaves the background uncovered)
         */
        glm::vec3 clear_color = glm::vec3(0.02f, 0.02f, 0.03f);

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
         * @note the overlay is drawn inside render_frame() (and the split frame steps): its
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
         * @brief drive one application frame: poll window events, respond to ESC / native close,
         *        skip rendering while the window is minimized, recreate the swapchain on restore
         *        and on resize (VK_ERROR_OUT_OF_DATE_KHR / VK_SUBOPTIMAL_KHR), then record,
         *        submit and present one frame when renderable
         * @return frame_status: proceed when a frame was presented; skipped when not renderable
         *         (minimized or swapchain recreated — caller yields and calls again); closed on
         *         window close; one of the stage-specific *_failed values on a fatal Vulkan error
         *         (caller exits the loop; is_failure() tests for any of them)
         * @note convenience wrapper that calls the split frame steps below in order
         *       (is_skipable -> try_recreate_swap_chain_if_minimized ->
         *       set_up_frame_environment -> begin_recording -> record_main_drawcalls ->
         *       end_recording -> submit_and_present). External code may call those steps itself
         *       to interleave custom recording (e.g. a debug overlay) between the steps.
         */
        frame_status render_frame();

        /**
         * @ingroup vulkan_runtime
         * @brief step 1 of the frame: poll window events and decide whether this iteration can
         *        render at all
         * @return frame_status::closed when the window was closed (ESC or native close);
         *         frame_status::skipped when the window is minimized (rendering would fail);
         *         frame_status::proceed when the caller may continue the frame
         * @note part of the split render_frame(); see render_frame() for the full sequence
         */
        frame_status is_skipable();

        /**
         * @ingroup vulkan_runtime
         * @brief step 2 of the frame: if the window was minimized since the last rendered frame,
         *        recreate the swapchain (its extent is 0-sized while minimized). Call only after
         *        is_skipable() reported proceed.
         * @note part of the split render_frame(); see render_frame() for the full sequence
         */
        void try_recreate_swap_chain_if_minimized();

        /**
         * @ingroup vulkan_runtime
         * @brief step 3 of the frame: wait the frame slot's fence, acquire the next swapchain
         *        image (recreating the swapchain when it is out of date) and write the shared
         *        camera UBO for this frame
         * @return frame_status::skipped when the swapchain was recreated (caller yields and
         *         retries next iteration); frame_status::acquire_failed when acquiring the image
         *         failed (other than out-of-date); frame_status::proceed when a frame may be recorded
         * @note part of the split render_frame(); see render_frame() for the full sequence
         */
        frame_status set_up_frame_environment();

        /**
         * @ingroup vulkan_runtime
         * @brief step 4 of the frame: begin recording the frame slot's command buffer and run
         *        the CPU-side scene prep (world-matrix accumulation + frustum culling)
         * @return frame_status::begin_recording_failed when vkBeginCommandBuffer failed
         *         (caller exits the loop); frame_status::proceed otherwise
         * @note part of the split render_frame(); see render_frame() for the full sequence
         */
        frame_status begin_recording();

        /**
         * @ingroup vulkan_runtime
         * @brief step 5 of the frame: record the runtime's own draw calls — the shadow pass,
         *        the attachment transitions and the main scene pass (skybox + visible leaves).
         *        The main rendering instance is left OPEN on purpose so an external caller can
         *        append extra draws (e.g. an ImGui overlay) into the same pass afterwards.
         * @note part of the split render_frame(); see render_frame() for the full sequence.
         *       Callers appending draws must end the rendering instance themselves via
         *       end_recording() (or vkCmdEndRendering before it, if they opened their own).
         */
        void record_main_drawcalls();

        /**
         * @ingroup vulkan_runtime
         * @brief the command buffer currently being recorded (between begin_recording() and
         *        end_recording()); external code may record additional draws into it after
         *        record_main_drawcalls() while the main rendering instance is still open
         */
        [[nodiscard]] VkCommandBuffer active_command_buffer() const noexcept;

        /**
         * @ingroup vulkan_runtime
         * @brief step 6 of the frame: end the main rendering instance (or the classic render
         *        pass), transition the swapchain image to PRESENT_SRC (dynamic rendering only)
         *        and finish recording the command buffer
         * @return frame_status::end_recording_failed when vkEndCommandBuffer failed (caller
         *         exits the loop); frame_status::proceed otherwise
         * @note part of the split render_frame(); see render_frame() for the full sequence
         */
        frame_status end_recording();

        /**
         * @ingroup vulkan_runtime
         * @brief step 7 of the frame: submit the recorded command buffer and present the
         *        swapchain image, recreating the swapchain when presentation reports out of date
         * @return frame_status::proceed when the frame was presented;
         *         frame_status::submit_failed when vkQueueSubmit failed;
         *         frame_status::present_failed when presentation failed (other than out-of-date)
         * @note part of the split render_frame(); see render_frame() for the full sequence
         */
        frame_status submit_and_present();

        std::expected<void, std::string> make_pipeline(
            std::string_view pipeline_name,
            std::span<unsigned char const> vertex_shader_code,
            std::span<unsigned char const> fragment_shader_code);

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
         * @note requires dynamic rendering (Vulkan 1.3); on the classic render-pass fallback
         *       path the creation fails and shadow mapping stays disabled
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
         * @brief access the scene tree (roots + children + per-node local transforms) for
         *        programmatic whole-group / subtree transforms
         * @note the tree structure is fixed after import (no reallocation of scene or the
         *       children vectors while nodes are only edited in place), so pointers/references
         *       into the tree stay valid until the next make_primitive / import / clear call
         */
        [[nodiscard]] scene_tree::scene& get_scene() noexcept {
            return this->scene;
        }
        [[nodiscard]] scene_tree::scene const& get_scene() const noexcept {
            return this->scene;
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
         */
        void set_ibl(ibl_input const& info);

        /**
         * @ingroup vulkan_runtime
         * @brief create a primitive and attach it to the scene tree as a new leaf (root node).
         * @param pipeline_name the pipeline the primitive draws with (must already exist)
         * @param info geometry and material textures
         * @return pointer to the created primitive (owned by the scene tree), or nullptr if the
         *         pipeline does not exist
         * @note the leaf's local transform is @p info.model_matrix and its world is identity
         *       (render_frame runs update_world before drawing, so primitive::set_world writes
         *       the same matrix into push.model as before)
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
         */
        template <class NI, class NS, class DI, class DS>
            requires scene_node_iterator<NI> && std::same_as<NS, NI> && scene_drawable_iterator<DI> && std::same_as<DS, DI>
        scene_import_result import_scene(NI nfirst, NS nlast, DI dfirst, DS dlast, glm::vec3 const& offset) {
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
                info.emissive = to_texture(drawable.get_emissive(), VK_FORMAT_R8G8B8A8_UNORM);
                auto const factors = drawable.get_factors();
                info.factors.base_color_factor = factors.base_color_factor;
                info.factors.emissive_factor = factors.emissive_factor;
                info.factors.metallic_factor = factors.metallic_factor;
                info.factors.roughness_factor = factors.roughness_factor;
                info.factors.normal_scale = factors.normal_scale;
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
                node.local = nfirst.get_local_transform();
                if (depth == 0) {
                    node.local = glm::translate(glm::mat4(1.0f), offset) * node.local; // scene root gets the offset
                }
                // attach under the parent (depth-1) or as a new scene root
                if (depth == 0) {
                    this->scene.roots.push_back(std::move(node));
                    ancestors.assign(1, &this->scene.roots.back());
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