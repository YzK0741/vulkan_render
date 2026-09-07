module;

#include <glm/glm.hpp>

export module vulkan.animation;

import std;
import gltf_loader;

/**
 * @file animation_controller.cppm
 * @defgroup vulkan_animation Vulkan Animation Controller
 * @brief bridge between glTF animation/skin/morph data (gltf_loader, pure CPU) and a scene
 *        runtime: plays the file's keyframe animations by sampling pure CPU, writing the
 *        evaluated T/R/S back into scene node locals, and rebuilding the per-frame skin
 *        matrices + morph weights into the host's per-slot buffers.
 *
 * The controller never depends on the concrete host class (vulkan::runtime) NOR on the
 * host's scene-tree types (scene_node / primitive / push constants): it talks to whatever
 * owns the scene through an injected animation_backend (opaque node handles, plain-data
 * snapshots, numeric capacities and callbacks for the few mutations it performs), so the
 * same controller can drive any host that provides the same surface - gltf_loader is the
 * only other dependency (the animation data being played). This module is the only place
 * that knows how glTF keyframes/skins/morph targets map onto a scene.
 *
 * Contract summary:
 *   - init() resolves the skin/morph rigs against the backend's scene and bakes static
 *     data (identity skin block, morph deltas) into every frame slot's buffers; call it
 *     before the first frame, or only while the host is idle.
 *   - update(dt) writes the paced frame slot's skin/morph buffers and scene node locals, so
 *     call it after the host paced a frame slot and before it records (after the slot's
 *     timeline wait).
 */
namespace vulkan {
    /**
     * @ingroup vulkan_animation
     * @brief one node of the host scene as the controller sees it: an opaque handle plus the
     *        identity bits animation needs (source node index, root flag). The host hands out
     *        a snapshot of these at init(); handles stay valid while the tree is frozen.
     */
    export struct scene_node_info {
        std::uint64_t id = 0;         // opaque handle; only meaningful to the backend
        std::size_t source_index = 0; // asset node index this node was rebuilt from
        bool is_root = false;         // scene root (import shift re-applied on local writes)
    };

    /**
     * @ingroup vulkan_animation
     * @brief the host surface an animation_controller drives, injected at init(). Everything
     *        the controller needs from the host, expressed without host types: opaque node
     *        handles, plain-data snapshots, numeric capacities, and callbacks for the few
     *        mutations it performs (node locals, skin/morph block tags on the mesh leaves,
     *        per-frame joint world collection). The host wires these to its scene tree +
     *        frame-slot buffers + worker pool (see chores).
     */
    export struct animation_backend {
        // ---- capacities / constants (host provides; controller checks against them) ----
        std::size_t morph_capacity = 0; // floats per frame-slot morph buffer
        uint32_t skin_capacity = 0;     // mat4s per frame-slot skin buffer
        uint32_t frames_in_flight = 0;  // frame slots (identity block baked into each)

        // ---- scene tree access (opaque handles; valid while the tree stays frozen) ----
        std::function<std::vector<scene_node_info>()> snapshot_nodes;        // all nodes, DFS pre-order
        std::function<void(std::uint64_t, glm::mat4 const&)> set_node_local; // write one node's local
        std::function<std::string_view(std::uint64_t)> node_name;            // debug labels

        // ---- skin / morph block tagging (setup-time; host knows its leaf types) ----
        // point every leaf of the skinned mesh node (its own leaf + "/prim" extra leaves) at
        // @p block_base in the skin buffer
        std::function<void(std::size_t mesh_source, uint32_t block_base)> assign_skin_block;
        // leaves of one source with their vertex counts (for morph layout; "/prim" extras
        // inherit their parent's source - resolved by the host)
        std::function<std::vector<std::pair<std::uint64_t, uint32_t>>(std::size_t source)> morph_leaves;
        // tag one morphable leaf's morph block start / target count / vertex count
        std::function<void(std::uint64_t, uint32_t morph_base, uint32_t targets, uint32_t vertices)> set_morph_block;
        // world matrices of the given asset node indices this frame (host walks its tree)
        std::function<std::unordered_map<std::size_t, glm::mat4>(std::span<std::size_t const> sources)> collect_worlds;

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

        /** @brief whether the backend is fully wired (snapshot + per-frame essentials) */
        [[nodiscard]] bool valid() const noexcept {
            return this->snapshot_nodes && this->set_node_local && this->morph_scratch_active &&
                   this->set_skin_matrices_active && this->run_tasks;
        }
    };

    /**
     * @ingroup vulkan_animation
     * @brief one glTF animation + its playback state, applied to the host scene
     */
    export class animation_controller {
    public:
        /**
         * @ingroup vulkan_animation
         * @brief build the playback table and resolve the skin/morph rigs against the backend's
         *        scene: collect the playable (channel-bearing) animations and TRS base poses from
         *        the loader, map asset node indices onto live scene nodes (via the backend's
         *        snapshot), bake the morph deltas with their default weights into every frame
         *        slot's morph buffer and upload the identity skin block into every slot's skin
         *        buffer. Skinned/morphable leaves get their block tags through the backend
         *        (assign_skin_block / set_morph_block).
         * @param scenes the loaded glTF data (loader node pool + animations + skins + meshes)
         * @param backend the host surface to drive (snapshot + callbacks; see animation_backend)
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
         * @note call after the host paced a frame slot and before it records (the host's
         *       per-slot buffers may only be written once the slot is paced)
         */
        void update(float dt_seconds);

        // ---- read-only bridge for demo-side consumers (e.g. glTF camera seeding) ----

        /** @brief loader nodes by asset node index (kept alive by @p scenes) */
        [[nodiscard]] std::unordered_map<std::size_t, gltf::node const*> const& get_loader_nodes() const noexcept;
        /** @brief whether an asset node index occurs in the host scene */
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
            std::uint64_t node = 0; // opaque backend handle of a scene node
            bool scene_root = false;
        };
        struct skin_rig {
            gltf::skin const* skin = nullptr; // loader skin: joints (asset node indices) + IBM
            std::size_t mesh_source = 0;      // asset node index of the skinned mesh node
            uint32_t block_base = 0;          // block start in the skin buffer (after identity)
        };
        struct morph_rig {
            std::size_t source = 0; // owning loader node (weights animation target)
            uint32_t vertex_count = 0;
            uint32_t target_count = 0;
            uint32_t morph_base = 0; // float index into the morph buffer
        };

        animation_backend backend; // injected host surface; valid() == false when unbound
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
        // mesh node plus every joint it references. Fixed after init(); collected in one pass
        // through the backend each frame (skin_sources is an O(1) membership set).
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
