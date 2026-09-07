module;

#include <glm/glm.hpp>

export module vulkan.animation;

import std;
import gltf_loader;
import vulkan.runtime.scene_tree; // scene + node/primitive types (the one structural dependency)

/**
 * @file animation_controller.cppm
 * @defgroup vulkan_animation Vulkan Animation Controller
 * @brief bridge between glTF animation/skin/morph data (gltf_loader, pure CPU) and a scene
 *        runtime: plays the file's keyframe animations by sampling pure CPU, writing the
 *        evaluated T/R/S back into scene node locals, and rebuilding the per-frame skin
 *        matrices + morph weights into the host's per-slot buffers.
 *
 * The controller never depends on the concrete host class (vulkan::runtime): it talks to
 * whatever owns the scene through an injected animation_backend (callbacks + a scene&), so
 * the same controller can drive any host that provides the same surface. This module is the
 * only place that knows how glTF keyframes/skins/morph targets map onto a scene tree.
 *
 * Contract summary (mirrors make_primitive / import_scene / set_ibl):
 *   - init() registers materials/geometry state and writes every scene set's shared buffers,
 *     so call it before the first frame, or only while the runtime is idle.
 *   - update(dt) writes the paced frame slot's skin/morph buffers and scene node locals, so
 *     call it after the host paced a frame slot and before it records (after the slot's
 *     timeline wait).
 */
namespace vulkan {
    /**
     * @ingroup vulkan_animation
     * @brief the host surface an animation_controller drives, injected at init(): the scene
     *        tree it mutates plus callbacks for everything else it needs from the host.
     *
     * Kept deliberately narrow: only what per-frame glTF playback touches. The scene is a
     * direct reference (animation must walk and edit nodes in place); the rest are callbacks
     * so the controller does not depend on the host class - any object exposing the same
     * surface can drive animations. assemble via the host side (see chores).
     */
    export struct animation_backend {
        vulkan::scene_tree::scene* scene = nullptr; // tree to animate (nullptr = not bound)

        // ---- per-frame (active slot) access, used by update() ----
        std::function<float*()> morph_scratch_active;                             // host-visible morph scratch of the paced slot
        std::function<void(std::span<glm::mat4 const>)> set_skin_matrices_active; // upload skin matrices to the paced slot

        // ---- setup-time (explicit slot) access, used by init() ----
        std::function<float*(uint32_t)> morph_scratch_slot;                               // morph scratch of one frame slot
        std::function<void(std::span<glm::mat4 const>, uint32_t)> set_skin_matrices_slot; // upload to one frame slot

        // ---- frame-loop cooperation ----
        std::function<void()> scene_changed;                             // node locals edited -> caller invalidates caches
        std::function<void(std::span<std::function<void()>>)> run_tasks; // fan tasks out on the host's worker pool (sync)
        std::function<int()> task_worker_count;                          // pool workers, for slicing fan-out tasks
    };

    /**
     * @ingroup vulkan_animation
     * @brief one glTF animation + its playback state, applied to the scene tree
     */
    export class animation_controller {
    public:
        /**
         * @ingroup vulkan_animation
         * @brief build the playback table and resolve the skin/morph rigs against the backend's
         *        scene: collect the playable (channel-bearing) animations and TRS base poses from
         *        the loader, map asset node indices onto live scene nodes, bake the morph deltas
         *        with their default weights into every frame slot's morph buffer and upload the
         *        identity skin block into every slot's skin buffer. Skinned/morphable primitives
         *        get their push.skin_base / push.morph_* fields set here.
         * @param scenes the loaded glTF data (loader node pool + animations + skins + meshes)
         * @param backend the host surface to drive (scene + per-slot callbacks; see animation_backend)
         * @param import_shift translation the import applied to every scene ROOT node's local
         *        (animated roots must re-apply it, like import_scene did)
         * @note call before the first frame, or only while the host is idle (no frame in
         *       flight) - this writes scene buffers/descriptors like make_primitive() does.
         */
        void init(gltf::scenes const& scenes, animation_backend const& backend, glm::vec3 const& import_shift);

        // ---- playback table / gui binding ----

        /** @brief number of channel-bearing animations (the combo lists these) */
        [[nodiscard]] std::size_t playable_count() const noexcept;
        /** @brief display name of playable @p index ("<unnamed>" when the glTF has none) */
        [[nodiscard]] std::string_view playable_name(std::size_t index) const noexcept;
        /** @brief longest playable duration (fixed slider range, like the old demo combo) */
        [[nodiscard]] float playable_max_duration() const noexcept;
        /** @brief true when a playable animation is selected (auto-picks the first on init) */
        [[nodiscard]] bool has_active() const noexcept;
        /** @brief index of the active playable in the playable list */
        [[nodiscard]] std::size_t current() const noexcept;
        /** @brief switch to playable @p index: reset every animated node to its base pose (so
         *         nodes the previous animation moved but the new one does not return), then set
         *         time to zero. Keeps the playing flag as-is.
         */
        void select(std::size_t index);
        /** @brief pause/resume the clock (sampling keeps running while paused) */
        void set_playing(bool playing) noexcept;
        /** @brief whether the clock advances each update() */
        [[nodiscard]] bool is_playing() const noexcept;
        /** @brief scrub to @p t seconds and pause (mirrors the gui time-slider behavior: the
         *         next update() samples the new time without the clock fighting the drag) */
        void set_time(float t);
        /** @brief current playback time in seconds */
        [[nodiscard]] float current_time() const noexcept;
        /** @brief loop length of the active animation in seconds */
        [[nodiscard]] float loop_duration() const noexcept;

        /**
         * @ingroup vulkan_animation
         * @brief advance and apply one frame: sample the active animation at the (possibly
         *        advanced) time, write each animated node's T/R/S local (scene roots keep the
         *        import shift) and mark the scene changed, write the active frame slot's morph
         *        weights, then rebuild + upload the skin matrices into the active slot.
         * @param dt_seconds clock advance when playing (e.g. frame_clock::delta_seconds())
         * @note call after runtime.pace_and_acquire() and before runtime.begin_recording(): the runtime's
         *       per-slot buffers may only be written once pace_and_acquire() paced the slot.
         */
        void update(float dt_seconds);

        // ---- read-only bridge for demo-side consumers (e.g. glTF camera seeding) ----

        /** @brief loader nodes by asset node index (kept alive by @p scenes) */
        [[nodiscard]] std::unordered_map<std::size_t, gltf::node const*> const& get_loader_nodes() const noexcept;
        /** @brief whether an asset node index occurs in the runtime scene tree */
        [[nodiscard]] bool has_runtime_node(std::size_t source) const noexcept;

        // ---- per-second diagnostics (demo log lines) ----

        /** @brief display name of the active animation ("" when none) */
        [[nodiscard]] std::string_view active_name() const noexcept;
        /** @brief name of the reported animated node ("" when none) */
        [[nodiscard]] std::string_view get_debug_node_name() const noexcept;
        /** @brief translation of the reported animated node this frame */
        [[nodiscard]] glm::vec3 get_debug_translation() const noexcept;
        /** @brief true when the first skin rig's last joint world was resolved this frame */
        [[nodiscard]] bool is_skin_debug_valid() const noexcept;
        /** @brief world x-axis of the first skin rig's LAST joint (rotation debug) */
        [[nodiscard]] glm::vec3 get_skin_debug_translation() const noexcept;
        /** @brief display name of the first active skin rig ("" when none) */
        [[nodiscard]] std::string_view get_skin_debug_name() const noexcept;

    private:
        struct anim_target {
            vulkan::scene_tree::scene_node* node = nullptr;
            bool scene_root = false;
        };
        struct skin_rig {
            gltf::skin const* skin = nullptr; // loader skin: joints (asset node indices) + IBM
            std::size_t mesh_source = 0;      // asset node index of the skinned mesh node
            uint32_t block_base = 0;          // block start in the skin buffer (after identity)
        };
        struct morph_rig {
            vulkan::primitive* prim = nullptr;
            uint32_t vertex_count = 0;
            uint32_t target_count = 0;
            uint32_t morph_base = 0; // float index into the morph buffer
            std::size_t source = 0;  // owning loader node (weights animation target)
        };

        animation_backend backend; // injected host surface (scene + callbacks); scene == nullptr when unbound
        glm::vec3 import_shift{};
        // whether this scene's animation is heavy enough to fan sampling out over the backend's
        // shared task pool (many channels over many sources): decided in init(), used by update()
        bool parallel_sampling = false;
        // source keys in stable order for parallel sampling (the source set is fixed after
        // init(); sample_keys mirrors source_nodes's keys so update() can slice them)
        std::vector<std::size_t> sample_keys = {};
        std::vector<gltf::animation const*> playable = {};
        std::unordered_map<std::size_t, std::vector<anim_target>> source_nodes = {};
        std::unordered_map<std::size_t, gltf::node_pose> base_poses = {};
        std::unordered_map<std::size_t, gltf::node const*> loader_nodes = {};
        gltf::animation const* active = nullptr;
        std::size_t current_index = 0;
        float time = 0.0f;
        float duration = 1.0f;
        float max_duration = 1.0f;
        bool playing = true;
        std::size_t debug_source = std::numeric_limits<std::size_t>::max();
        glm::vec3 debug_translation{};
        std::string debug_node_name = {};
        std::vector<skin_rig> skin_rigs = {};
        std::vector<morph_rig> morph_rigs = {};
        bool skin_debug_valid = false;
        glm::vec3 skin_debug_translation{};
        std::string skin_debug_name = {};
        // asset node indices whose world matrix update() must collect each frame: every rig's
        // mesh node plus every joint it references. Fixed after init(); the per-frame DFS only
        // tests each visited node against this set (O(1) contains) instead of scanning every
        // rig x joint pair per node.
        std::unordered_set<std::size_t> skin_sources = {};

        // the node reported per second: prefer a translation channel target, fall back to the
        // first channel target present in the tree
        std::size_t pick_debug_source(gltf::animation const& animation) const;
        void refresh_debug_name();

        // sample one loader source into its scene nodes + the active slot's morph weights at
        // this->time; returns whether any node local moved (morph-only writes are not
        // "changed": they do not invalidate the culling BVH). A member function so the
        // sampling fan-out tasks only capture `this` (+ their source range): the task list is
        // self-contained and can be handed to the backend's run_tasks for pool execution.
        bool sample_source(std::size_t source, std::vector<anim_target> const& targets);
    };
} // namespace vulkan
